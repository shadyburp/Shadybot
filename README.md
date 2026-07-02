# Telegram Solana Bot

Commands:
- `/pft` — pump.fun trending tokens (falls back to DexScreener if pump.fun blocks the request)
- Paste a **contract address** — get price/liquidity, top-holder %, and a bundle/sniper heuristic
- `/alert <mint> <above|below> <price_usd>` — price alert, checked every 60s
- `/alerts` — list your alerts
- `/unalert <number>` — remove one
- Anything else you type → answered by Claude

## Setup

1. **Get a Telegram bot token**
   - Message [@BotFather](https://t.me/BotFather) on Telegram
   - Send `/newbot`, follow the prompts, copy the token it gives you

2. **Get an Anthropic API key** (for the Q&A feature)
   - Go to https://console.anthropic.com/ → API Keys → Create Key
   - This is a *separate* thing from your claude.ai login — it's pay-as-you-go API billing

3. **Install Python 3.10+**, then:
   ```bash
   cd telegram_bot
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Configure secrets**
   ```bash
   cp .env.example .env
   ```
   Open `.env` and paste in your `TELEGRAM_BOT_TOKEN` and `ANTHROPIC_API_KEY`.

5. **Run it**
   ```bash
   python bot.py
   ```
   Leave this running (or deploy it to a small VPS / Railway / a Raspberry Pi) — it needs to stay running to respond.

## Honest limitations (read before you trust the output)

- **Bundle detection is a heuristic**, not a certified bundle count. It checks how many of a
  token's earliest on-chain transactions landed in the exact same Solana slot as launch — a big
  cluster there is a real signal of atomic/bundled buys, but it can't tell you *who* funded those
  wallets or give a precise bundle %. That needs a paid indexer (Helius, Solscan Pro, Bubblemaps)
  decoding full transaction graphs — happy to wire one in if you get a key.
- **Holder addresses aren't labeled.** You'll see raw wallet addresses and their % of supply, not
  "this one's a CEX wallet" or "this one's the LP." Labeling needs a paid service too.
- **No influencer/Twitter-mention tracking.** The bot only shows the project's own linked Twitter
  from DexScreener, not "which known accounts are tweeting about this." That's a genuinely
  different (and paid) data source — X API, TweetScout, Kolscan, etc.
- **pump.fun's trending endpoint may require auth** going forward (they've been locking down
  frontend API access) — the bot falls back to DexScreener's boosted-tokens list if so.
- **Public Solana RPC is rate-limited.** Fine for personal use; if you're hammering it you'll want
  your own RPC (Helius/QuickNode have free tiers).

None of this is financial advice — always verify before trading on it.
