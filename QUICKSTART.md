# Quick Start Guide

Get your Discord LLM Summarization Bot running in 5 minutes!

## Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] Discord Bot Token (see below)
- [ ] OpenAI API Key (from platform.openai.com)

## Step 1: Get Discord Bot Token

1. Go to https://discord.com/developers/applications
2. Click "New Application" → Give it a name
3. Go to "Bot" → Click "Add Bot"
4. **Enable these intents:**
   - ✅ Message Content Intent (REQUIRED!)
   - ✅ Server Members Intent
5. Click "Reset Token" → Copy your token

## Step 2: Get OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy your key immediately (you won't see it again!)

## Step 3: Setup Project

### Linux/Mac:
```bash
chmod +x setup.sh
./setup.sh
```

### Windows:
```batch
setup.bat
```

### Or manually:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

pip install -r requirements.txt
cp .env.example .env
```

## Step 4: Configure

Edit `.env` file:
```env
DISCORD_TOKEN=your_bot_token_here
OPENAI_API_KEY=your_openai_key_here
```

## Step 5: Invite Bot to Your Server

1. Go to Discord Developer Portal → Your App → OAuth2 → URL Generator
2. Select scopes: `bot` and `applications.commands`
3. Select permissions:
   - Read Messages/View Channels
   - Send Messages
   - Read Message History
   - Use Slash Commands
4. Copy URL → Open in browser → Select your server

## Step 6: Run!

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Run the bot
python bot.py
```

You should see:
```
Database initialized
Slash commands synced
YourBot#1234 has connected to Discord!
```

## Step 7: Test It Out!

In your Discord server:

1. Send some messages in a channel
2. Type `/catchup` to test
3. Or try `/summarize mode:2h` to summarize last 2 hours

## Common Commands

```bash
/catchup                                      # Quick summary since last seen
/catchup language:Indonesian                  # Quick summary in Indonesian
/summarize mode:2h                            # Summarize last 2 hours
/summarize mode:2h language:Indonesian        # Summarize in Indonesian
/summarize mode:1d detail:detailed            # Detailed summary of last day
/summarize mode:12h context:meeting           # Context-based summary
/summary_help                                 # Show all commands
```

## Troubleshooting

**Bot doesn't respond:**
- Check Message Content Intent is enabled in Discord
- Verify bot has permissions in the channel
- Check console for errors

**"Module not found" error:**
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt` again

**OpenAI errors:**
- Verify API key is correct
- Check you have credits at platform.openai.com/account/billing

## What's Next?

- Read full [README.md](README.md) for detailed documentation
- Customize settings in `.env`
- Modify prompts in `utils/llm_handler.py`
- Change LLM model for better quality or lower cost

## Cost Estimate

Using default `gpt-4o-mini`:
- ~$0.01-0.03 per 100-message summary
- Very affordable for personal/small server use
- Monitor usage at platform.openai.com/usage

---

**Need help?** Check README.md or open an issue on GitHub.
