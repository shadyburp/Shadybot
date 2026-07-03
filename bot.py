"""
Telegram bot: pump.fun trending, token lookups (holder distro + best-effort
bundle detection), price alerts, and AI Q&A fallback via Claude.

Setup: see README.md
"""

import asyncio
import io
import logging
import os
import re
from dataclasses import dataclass, field

import httpx
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import imagehash
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
log = logging.getLogger("bot")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
ALERT_POLL_SECONDS = int(os.getenv("ALERT_POLL_SECONDS", "60"))

SOL_ADDRESS_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")


# ----------------------------------------------------------------------------
# In-memory alert storage: {chat_id: [Alert, ...]}
# ----------------------------------------------------------------------------
@dataclass
class Alert:
    mint: str
    target_price: float
    direction: str  # "above" or "below"
    last_price: float = None


ALERTS: dict[int, list[Alert]] = {}


# ----------------------------------------------------------------------------
# Helpers: data sources
# ----------------------------------------------------------------------------
async def fetch_pumpfun_trending(limit: int = 10) -> list[dict]:
    """Try pump.fun's frontend API first, fall back to DexScreener boosted
    Solana tokens if pump.fun blocks the unauthenticated request."""
    url = (
        "https://frontend-api-v3.pump.fun/coins/currently-live"
        f"?offset=0&limit={limit}&includeNsfw=false&order=DESC&sort=market_cap"
    )
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data:
                    return [
                        {
                            "name": c.get("name"),
                            "symbol": c.get("symbol"),
                            "mint": c.get("mint"),
                            "market_cap": c.get("market_cap"),
                            "source": "pump.fun",
                        }
                        for c in data
                    ]
        except Exception as e:
            log.warning("pump.fun trending failed: %s", e)

        # Fallback: DexScreener boosted tokens, filtered to Solana
        try:
            r = await client.get(
                "https://api.dexscreener.com/token-boosts/top/v1",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            r.raise_for_status()
            data = r.json()
            sol = [d for d in data if d.get("chainId") == "solana"][:limit]
            return [
                {
                    "name": d.get("description", "")[:40] or d.get("tokenAddress"),
                    "symbol": "",
                    "mint": d.get("tokenAddress"),
                    "market_cap": None,
                    "source": "dexscreener (boosted, fallback)",
                }
                for d in sol
            ]
        except Exception as e:
            log.warning("dexscreener fallback failed: %s", e)
            return []


async def fetch_dexscreener_token(mint: str) -> dict | None:
    url = f"https://api.dexscreener.com/latest/dex/tokens/{mint}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            if r.status_code != 200:
                return None
            data = r.json()
            pairs = data.get("pairs") or []
            if not pairs:
                return None
            # pick the highest-liquidity pair
            pairs.sort(key=lambda p: (p.get("liquidity") or {}).get("usd", 0) or 0, reverse=True)
            return pairs[0]
    except Exception as e:
        log.warning("dexscreener fetch failed for %s: %s", mint, e)
        return None


async def solana_rpc(method: str, params: list) -> dict:
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            SOLANA_RPC_URL,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        )
        r.raise_for_status()
        return r.json()


async def fetch_holder_distribution(mint: str, top_n: int = 10) -> list[dict] | None:
    """Top holders by % of supply, using public RPC. No labels for
    CEX/LP wallets since that requires a paid indexer."""
    try:
        largest = await solana_rpc("getTokenLargestAccounts", [mint])
        supply_resp = await solana_rpc("getTokenSupply", [mint])
        total = float(supply_resp["result"]["value"]["amount"])
        if total == 0:
            return None
        accounts = largest["result"]["value"][:top_n]
        out = []
        for a in accounts:
            amt = float(a["amount"])
            out.append({"address": a["address"], "pct": round(100 * amt / total, 2)})
        return out
    except Exception as e:
        log.warning("holder distro fetch failed for %s: %s", mint, e)
        return None


async def fetch_bundle_signal(mint: str, sample_size: int = 40) -> dict:
    """Best-effort bundle/sniper heuristic: pulls the earliest transactions
    touching the mint and checks how many landed in the exact same slot.
    A large cluster of buys in one slot is a strong signal of a bundled
    launch (many wallets buying atomically via one bundler). This is a
    heuristic, not a definitive bundle count -- for that you'd want a paid
    indexer (Helius, Solscan Pro, Bubblemaps) that decodes full tx graphs.
    """
    try:
        sigs_resp = await solana_rpc(
            "getSignaturesForAddress", [mint, {"limit": sample_size}]
        )
        sigs = sigs_resp.get("result", [])
        if not sigs:
            return {"available": False}
        # earliest first
        sigs = list(reversed(sigs))
        slots = [s["slot"] for s in sigs if s.get("slot") is not None]
        if not slots:
            return {"available": False}
        first_slot = slots[0]
        same_slot_count = sum(1 for s in slots if s == first_slot)
        near_slot_count = sum(1 for s in slots if abs(s - first_slot) <= 2)
        return {
            "available": True,
            "sampled": len(slots),
            "same_slot_as_launch": same_slot_count,
            "within_2_slots_of_launch": near_slot_count,
        }
    except Exception as e:
        log.warning("bundle signal fetch failed for %s: %s", mint, e)
        return {"available": False}


async def dexscreener_search(query: str) -> list[dict]:
    """Search DexScreener across all chains for a query (symbol/name), filtered to Solana."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                "https://api.dexscreener.com/latest/dex/search", params={"q": query}
            )
            r.raise_for_status()
            data = r.json()
            pairs = data.get("pairs") or []
            return [p for p in pairs if p.get("chainId") == "solana"]
    except Exception as e:
        log.warning("dexscreener search failed for %s: %s", query, e)
        return []


async def _image_bytes(url: str) -> bytes | None:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            if r.status_code == 200:
                return r.content
    except Exception as e:
        log.warning("image fetch failed for %s: %s", url, e)
    return None


async def check_vamp(mint: str) -> dict:
    """Check whether a token looks like a copycat ('vamp') of an older,
    same-ticker Solana token: same symbol + older pair + (optionally) a
    visually similar logo. Heuristic, not definitive -- always DYOR.
    """
    target = await fetch_dexscreener_token(mint)
    if not target:
        return {"found": False}

    base = target.get("baseToken", {})
    symbol = base.get("symbol", "")
    target_created = target.get("pairCreatedAt")  # ms epoch, may be None
    target_image = (target.get("info") or {}).get("imageUrl")

    candidates = await dexscreener_search(symbol) if symbol else []
    older_matches = []

    target_hash = None
    if target_image:
        img_bytes = await _image_bytes(target_image)
        if img_bytes:
            try:
                target_hash = imagehash.phash(Image.open(io.BytesIO(img_bytes)))
            except Exception as e:
                log.warning("hash failed for target image: %s", e)

    for c in candidates:
        c_base = c.get("baseToken", {})
        if c_base.get("address") == base.get("address"):
            continue
        if c_base.get("symbol", "").lower() != symbol.lower():
            continue
        c_created = c.get("pairCreatedAt")
        # only count as "older" if we know both timestamps and this one predates target
        is_older = (
            target_created is not None and c_created is not None and c_created < target_created
        )
        if not is_older:
            continue

        visual_match = None
        c_image = (c.get("info") or {}).get("imageUrl")
        if target_hash and c_image:
            img_bytes = await _image_bytes(c_image)
            if img_bytes:
                try:
                    c_hash = imagehash.phash(Image.open(io.BytesIO(img_bytes)))
                    visual_match = (target_hash - c_hash) <= 8  # low hamming distance = similar
                except Exception:
                    pass

        older_matches.append(
            {
                "address": c_base.get("address"),
                "name": c_base.get("name"),
                "liquidity": (c.get("liquidity") or {}).get("usd"),
                "visual_match": visual_match,
            }
        )

    older_matches.sort(key=lambda m: m.get("liquidity") or 0, reverse=True)

    return {
        "found": True,
        "symbol": symbol,
        "name": base.get("name"),
        "older_matches": older_matches[:5],
    }


# ----------------------------------------------------------------------------
# Command handlers
# ----------------------------------------------------------------------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Yo. Here's what I do:\n\n"
        "/pft - pump.fun trending tokens\n"
        "Paste a contract address - I'll pull holder distro, a bundle/sniper "
        "signal, any linked Twitter/socials, plus refresh/chart buttons\n"
        "/pnl <mint> <entry_price> <amount> - generate a PnL card image\n"
        "/vamp <mint> - check for older copycat tokens with the same ticker\n"
        "/alert <mint> <above|below> <price_usd> - price alert\n"
        "/alerts - list your alerts\n"
        "/unalert <number> - remove one\n\n"
        "Anything else you type gets answered by Claude."
    )


async def pft_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("Pulling trending tokens...")
    coins = await fetch_pumpfun_trending(limit=10)
    if not coins:
        await msg.edit_text("Couldn't reach pump.fun or the fallback source right now. Try again shortly.")
        return
    lines = [f"*Trending* (source: {coins[0]['source']})\n"]
    for i, c in enumerate(coins, 1):
        mc = f"${c['market_cap']:,.0f}" if c.get("market_cap") else "n/a"
        name = c.get("name") or "unknown"
        sym = f" (${c['symbol']})" if c.get("symbol") else ""
        lines.append(f"{i}. {name}{sym} - MC {mc}\n`{c['mint']}`")
    await msg.edit_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def vamp_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /vamp <mint>")
        return
    mint = context.args[0]
    if not SOL_ADDRESS_RE.match(mint):
        await update.message.reply_text("That doesn't look like a valid Solana address.")
        return
    await send_vamp_report(update, mint)


async def send_vamp_report(update: Update, mint: str):
    msg = await update.message.reply_text("Scanning for copycats...")
    result = await check_vamp(mint)

    if not result.get("found"):
        await msg.edit_text("No live market found for that token yet, can't compare tickers.")
        return

    older = result["older_matches"]
    if not older:
        await msg.edit_text(
            f"*${result['symbol']}* — no older same-ticker Solana tokens found. "
            "Doesn't look like a copycat by this check, but always DYOR.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    lines = [f"⚠️ *${result['symbol']}* — found {len(older)} older token(s) with the same ticker:\n"]
    for m in older:
        liq = f"${m['liquidity']:,.0f} liq" if m.get("liquidity") else "liq unknown"
        visual = ""
        if m["visual_match"] is True:
            visual = " — 🖼️ *visually similar logo*"
        elif m["visual_match"] is False:
            visual = " — different logo"
        lines.append(f"`{m['address'][:6]}...{m['address'][-4:]}` ({liq}){visual}")
    lines.append(
        "\nOlder + higher liquidity + similar logo = classic copycat pattern. "
        "This is a heuristic, not proof — always DYOR."
    )
    await msg.edit_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


def build_token_keyboard(mint: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("🔄 Refresh", callback_data=f"refresh:{mint}"),
                InlineKeyboardButton("📊 Chart", url=f"https://dexscreener.com/solana/{mint}"),
            ],
            [InlineKeyboardButton("🧛 Check copycats", callback_data=f"vamp:{mint}")],
        ]
    )


async def build_token_report(mint: str) -> list[str]:
    try:
        dex_task = fetch_dexscreener_token(mint)
        holders_task = fetch_holder_distribution(mint)
        bundle_task = fetch_bundle_signal(mint)
        dex, holders, bundle = await asyncio.gather(dex_task, holders_task, bundle_task)
    except Exception as e:
        log.warning("lookup failed for %s: %s", mint, e)
        return ["Something went wrong pulling that token's data. Try again in a bit."]

    lines = []
    if dex:
        base = dex.get("baseToken", {})
        price = dex.get("priceUsd")
        liq = (dex.get("liquidity") or {}).get("usd")
        fdv = dex.get("fdv")
        lines.append(f"*{base.get('name')}* (${base.get('symbol')})")
        lines.append(f"Price: ${price}" if price else "Price: n/a")
        if liq:
            lines.append(f"Liquidity: ${liq:,.0f}")
        if fdv:
            lines.append(f"FDV: ${fdv:,.0f}")
        socials = (dex.get("info") or {}).get("socials", [])
        twitter = next((s["url"] for s in socials if s.get("type") == "twitter"), None)
        if twitter:
            lines.append(f"Twitter: {twitter}")
        else:
            lines.append("Twitter: none linked on DexScreener")
    else:
        lines.append("No DexScreener pair found (token may not have a live market yet, e.g. still bonding on pump.fun).")

    lines.append("")
    if holders:
        lines.append("*Top holders (% of supply):*")
        for h in holders[:10]:
            lines.append(f"`{h['address'][:6]}...{h['address'][-4:]}` - {h['pct']}%")
    else:
        lines.append("Holder data unavailable right now.")

    lines.append("")
    if bundle.get("available"):
        lines.append(
            f"*Bundle/sniper signal (heuristic, not exact):* of the first "
            f"{bundle['sampled']} txs, {bundle['same_slot_as_launch']} landed in "
            f"the exact same slot as launch, {bundle['within_2_slots_of_launch']} "
            f"within 2 slots. High same-slot counts suggest bundled/atomic buys."
        )
    else:
        lines.append("Bundle signal unavailable right now.")

    lines.append(
        "\n_Note: influencer/Twitter-mention tracking (who's talking about this "
        "token) needs a paid API like the X API - not wired up yet._"
    )
    return lines


async def handle_contract_address(update: Update, context: ContextTypes.DEFAULT_TYPE, mint: str):
    msg = await update.message.reply_text("Looking that up...")
    lines = await build_token_report(mint)
    keyboard = build_token_keyboard(mint)
    try:
        await msg.edit_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
    except Exception as e:
        log.warning("edit_text failed, retrying without markdown: %s", e)
        await msg.edit_text("\n".join(lines), reply_markup=keyboard)


async def refresh_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer("Refreshing...")
    mint = query.data.split(":", 1)[1]
    lines = await build_token_report(mint)
    keyboard = build_token_keyboard(mint)
    try:
        await query.edit_message_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
    except Exception as e:
        log.warning("refresh edit failed: %s", e)


async def vamp_button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer("Scanning for copycats...")
    mint = query.data.split(":", 1)[1]
    result = await check_vamp(mint)

    if not result.get("found"):
        await context.bot.send_message(query.message.chat_id, "No live market found for that token yet.")
        return

    older = result["older_matches"]
    if not older:
        await context.bot.send_message(
            query.message.chat_id,
            f"*${result['symbol']}* — no older same-ticker Solana tokens found. "
            "Doesn't look like a copycat by this check, but always DYOR.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    lines = [f"⚠️ *${result['symbol']}* — found {len(older)} older token(s) with the same ticker:\n"]
    for m in older:
        liq = f"${m['liquidity']:,.0f} liq" if m.get("liquidity") else "liq unknown"
        visual = ""
        if m["visual_match"] is True:
            visual = " — 🖼️ *visually similar logo*"
        elif m["visual_match"] is False:
            visual = " — different logo"
        lines.append(f"`{m['address'][:6]}...{m['address'][-4:]}` ({liq}){visual}")
    lines.append(
        "\nOlder + higher liquidity + similar logo = classic copycat pattern. "
        "This is a heuristic, not proof — always DYOR."
    )
    await context.bot.send_message(query.message.chat_id, "\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def alert_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) != 3 or args[1].lower() not in ("above", "below"):
        await update.message.reply_text("Usage: /alert <mint> <above|below> <price_usd>")
        return
    mint, direction, price_str = args
    try:
        price = float(price_str)
    except ValueError:
        await update.message.reply_text("Price has to be a number.")
        return
    if not SOL_ADDRESS_RE.match(mint):
        await update.message.reply_text("That doesn't look like a valid Solana address.")
        return

    chat_id = update.effective_chat.id
    ALERTS.setdefault(chat_id, []).append(Alert(mint=mint, target_price=price, direction=direction.lower()))
    await update.message.reply_text(f"Alert set: {mint[:6]}... {direction} ${price}")


async def alerts_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    alerts = ALERTS.get(chat_id, [])
    if not alerts:
        await update.message.reply_text("No active alerts.")
        return
    lines = [
        f"{i+1}. {a.mint[:6]}...{a.mint[-4:]} {a.direction} ${a.target_price}"
        for i, a in enumerate(alerts)
    ]
    await update.message.reply_text("\n".join(lines))


async def unalert_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    alerts = ALERTS.get(chat_id, [])
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Usage: /unalert <number> (see /alerts for numbers)")
        return
    idx = int(context.args[0]) - 1
    if 0 <= idx < len(alerts):
        removed = alerts.pop(idx)
        await update.message.reply_text(f"Removed alert for {removed.mint[:6]}...")
    else:
        await update.message.reply_text("No alert with that number.")


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Try a few common system font paths; fall back to PIL's built-in font
    if none are found (still renders, just less pretty)."""
    candidates = [
        "/system/fonts/Roboto-Bold.ttf",
        "/data/data/com.termux/files/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default(size=size)


def render_pnl_card(symbol: str, entry_price: float, current_price: float, amount: float) -> io.BytesIO:
    pnl_pct = ((current_price - entry_price) / entry_price) * 100
    pnl_usd = (current_price - entry_price) * amount
    is_profit = pnl_usd >= 0

    W, H = 900, 500
    bg = (14, 20, 18) if is_profit else (24, 14, 14)
    accent = (57, 217, 138) if is_profit else (232, 79, 79)

    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    title_font = _load_font(46)
    big_font = _load_font(64)
    label_font = _load_font(26)
    small_font = _load_font(22)

    draw.text((40, 35), f"${symbol.upper()}", font=title_font, fill=(255, 255, 255))

    sign = "+" if pnl_usd >= 0 else ""
    draw.text((40, 130), f"{sign}${pnl_usd:,.2f}", font=big_font, fill=accent)
    draw.text((40, 210), f"{sign}{pnl_pct:.2f}%", font=title_font, fill=accent)

    y = 320
    rows = [
        ("Entry price", f"${entry_price:.8f}".rstrip("0").rstrip(".")),
        ("Current price", f"${current_price:.8f}".rstrip("0").rstrip(".")),
        ("Amount", f"{amount:,.4f} {symbol.upper()}"),
    ]
    for label, value in rows:
        draw.text((40, y), label, font=small_font, fill=(150, 150, 150))
        draw.text((300, y), value, font=label_font, fill=(230, 230, 230))
        y += 45

    buf = io.BytesIO()
    buf.name = "pnl.png"
    img.save(buf, "PNG")
    buf.seek(0)
    return buf


async def pnl_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) != 3:
        await update.message.reply_text(
            "Usage: /pnl <mint> <entry_price_usd> <amount_of_tokens>\n"
            "Example: /pnl 7xKX...pump 0.0000045 1000000"
        )
        return
    mint, entry_str, amount_str = args
    if not SOL_ADDRESS_RE.match(mint):
        await update.message.reply_text("That doesn't look like a valid Solana address.")
        return
    try:
        entry_price = float(entry_str)
        amount = float(amount_str)
    except ValueError:
        await update.message.reply_text("Entry price and amount both need to be numbers.")
        return
    if entry_price <= 0 or amount <= 0:
        await update.message.reply_text("Entry price and amount both need to be greater than 0.")
        return

    msg = await update.message.reply_text("Building your PnL card...")
    dex = await fetch_dexscreener_token(mint)
    if not dex or not dex.get("priceUsd"):
        await msg.edit_text("Couldn't find a live price for that token right now.")
        return

    symbol = (dex.get("baseToken") or {}).get("symbol", "TOKEN")
    current_price = float(dex["priceUsd"])

    try:
        card = render_pnl_card(symbol, entry_price, current_price, amount)
        await update.message.reply_photo(photo=card)
        await msg.delete()
    except Exception as e:
        log.warning("pnl card render failed: %s", e)
        await msg.edit_text("Something went wrong generating the card.")


# ----------------------------------------------------------------------------
# Background price-alert polling
# ----------------------------------------------------------------------------
async def price_alert_loop(app: Application):
    while True:
        await asyncio.sleep(ALERT_POLL_SECONDS)
        for chat_id, alerts in list(ALERTS.items()):
            if not alerts:
                continue
            for alert in list(alerts):
                dex = await fetch_dexscreener_token(alert.mint)
                if not dex or not dex.get("priceUsd"):
                    continue
                price = float(dex["priceUsd"])
                triggered = (
                    (alert.direction == "above" and price >= alert.target_price)
                    or (alert.direction == "below" and price <= alert.target_price)
                )
                if triggered:
                    try:
                        await app.bot.send_message(
                            chat_id,
                            f"🔔 {alert.mint[:6]}...{alert.mint[-4:]} hit ${price:.6f} "
                            f"({alert.direction} ${alert.target_price})",
                        )
                    except Exception as e:
                        log.warning("failed to send alert: %s", e)
                    alerts.remove(alert)


# ----------------------------------------------------------------------------
# Q&A fallback + message router
# ----------------------------------------------------------------------------
async def ai_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    if not GROQ_API_KEY:
        await update.message.reply_text(
            "AI Q&A isn't set up yet — add GROQ_API_KEY to your .env to enable it."
        )
        return
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [{"role": "user", "content": text}],
                    "max_tokens": 800,
                },
            )
            r.raise_for_status()
            data = r.json()
            reply = data["choices"][0]["message"]["content"]
        await update.message.reply_text(reply or "No response generated.")
    except Exception as e:
        log.warning("groq call failed: %s", e)
        await update.message.reply_text("AI request failed, try again in a bit.")


async def message_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if SOL_ADDRESS_RE.match(text):
        await handle_contract_address(update, context, text)
    else:
        await ai_reply(update, context, text)


# ----------------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------------
async def post_init(app: Application):
    app.create_task(price_alert_loop(app))


def main():
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in your .env file first.")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", start_cmd))
    app.add_handler(CommandHandler("pft", pft_cmd))
    app.add_handler(CommandHandler("pnl", pnl_cmd))
    app.add_handler(CommandHandler("vamp", vamp_cmd))
    app.add_handler(CommandHandler("alert", alert_cmd))
    app.add_handler(CommandHandler("alerts", alerts_cmd))
    app.add_handler(CommandHandler("unalert", unalert_cmd))
    app.add_handler(CallbackQueryHandler(refresh_callback, pattern="^refresh:"))
    app.add_handler(CallbackQueryHandler(vamp_button_callback, pattern="^vamp:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_router))

    log.info("Bot starting...")
    app.run_polling()


if __name__ == "__main__":
    main()
