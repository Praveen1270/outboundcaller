# outboundcaller

Production-grade AI outbound voice calling platform — LiveKit + Gemini Live + Vobiz SIP + Supabase.

## Features

- **Gemini Live** real-time voice agent (`gemini-3.1-flash-live-preview`)
- **Outbound SIP** dialing via Vobiz
- **Appointment booking**, CRM, campaigns, agent profiles
- **Single-file dashboard** at `http://localhost:8000`
- **Supabase** for all persistence
- **Docker / Coolify** deployment

## Quick Start

```bash
cp .env.example .env
# Fill in credentials in .env

pip install -r requirements.txt
sh start.sh
```

Open **http://localhost:8000**

## One-Time Setup

1. Run `supabase_schema.sql` in Supabase SQL Editor
2. Configure `.env` (see `.env.example`)
3. Dashboard → **Settings** → save keys → **Create SIP Trunk**
4. **Single Call** → test with your number

## Project Structure

```
agent.py              LiveKit worker (Gemini Live)
server.py             FastAPI + APScheduler
db.py                 Supabase operations
tools.py              LLM function tools
prompts.py            System prompt template
ui/index.html         Dashboard (single file)
start.sh              Production startup
Dockerfile            Container build
supabase_schema.sql   Database schema
```

## Deploy (Coolify)

- Port: **8000**
- Set env vars from `.env.example`
- CMD: `sh start.sh`

## License

MIT
