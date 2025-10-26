# Discord LLM Chat Summarization Bot

A Discord bot that uses OpenAI's GPT models to summarize chat conversations, helping users catch up on messages they missed while offline.

## Features

- **Automatic Message Tracking**: Tracks all messages in channels where the bot has access
- **Smart Summarization**: Uses GPT-4o-mini for cost-effective, high-quality summaries
- **Multiple Summary Modes**:
  - Catchup mode: Summarize since you were last active
  - Time-based: Summarize last X hours/days (e.g., "2h", "1d")
  - Context-based: Focus on specific topics or keywords
- **Flexible Detail Levels**: Brief, General, or Detailed summaries
- **Multi-Language Support**: Summaries available in English and Indonesian (Bahasa Indonesia)
- **User Activity Tracking**: Automatically tracks when users are active
- **Message Retention**: Configurable cleanup of old messages to save space

## Prerequisites

- Python 3.8 or higher
- Discord Bot Token (with Message Content Intent enabled)
- OpenAI API Key

## Installation

### 1. Clone the repository or navigate to project directory

```bash
cd nlp-llmAgent-miawDuwa
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```env
DISCORD_TOKEN=your_discord_bot_token_here
OPENAI_API_KEY=your_openai_api_key_here
MAX_MESSAGES_TO_SUMMARIZE=200
MESSAGE_RETENTION_DAYS=7
SUMMARY_MAX_LENGTH=2000
```

## Discord Bot Setup

### 1. Create a Discord Application

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application" and give it a name
3. Go to the "Bot" section
4. Click "Add Bot"
5. **IMPORTANT**: Under "Privileged Gateway Intents", enable:
   - **Message Content Intent** (required to read messages)
   - Server Members Intent (recommended)
   - Presence Intent (optional)

### 2. Get your bot token

1. In the Bot section, click "Reset Token"
2. Copy the token and add it to your `.env` file

### 3. Invite the bot to your server

1. Go to the "OAuth2" → "URL Generator" section
2. Select scopes:
   - `bot`
   - `applications.commands`
3. Select bot permissions:
   - Read Messages/View Channels
   - Send Messages
   - Read Message History
   - Use Slash Commands
4. Copy the generated URL and open it in your browser
5. Select your server and authorize the bot

## Running the Bot

```bash
python bot.py
```

You should see:
```
Database initialized
Slash commands synced
YourBotName#1234 has connected to Discord!
Bot is in X guilds
```

## Usage

### Commands

#### `/catchup`
Quick bullet-point summary of messages since you were last active.

**Parameters:**
- `language` (optional): Summary language - English (default) or Indonesian

**Examples:**
```
/catchup
/catchup language:Indonesian
```

#### `/summarize`
Comprehensive summarization with multiple options.

**Parameters:**
- `mode` (required): How to summarize
  - `catchup` - Since you were last active (default)
  - `2h`, `30m`, `1d` - Time-based (last X hours/minutes/days)
  - Combined with `context` for topic-based summaries

- `context` (optional): Specific topic to focus on

- `detail` (optional): Summary detail level
  - `brief` - 2-3 sentence overview
  - `general` - Balanced summary (default)
  - `detailed` - Comprehensive with topics and participants

- `language` (optional): Summary language - English (default) or Indonesian

**Examples:**

```bash
# Summarize since you were last active
/summarize mode:catchup

# Summarize last 2 hours
/summarize mode:2h

# Detailed summary of last 24 hours
/summarize mode:1d detail:detailed

# Brief summary of last 3 hours
/summarize mode:3h detail:brief

# Summarize in Indonesian
/summarize mode:2h language:Indonesian

# Detailed Indonesian summary
/summarize mode:1d detail:detailed language:Indonesian

# Context-based: Find discussions about "meeting" in last 12 hours
/summarize mode:12h context:meeting

# Context-based in Indonesian
/summarize mode:12h context:meeting language:Indonesian

# Context-based: Find discussions about "project deadline" in last day
/summarize mode:1d context:project deadline detail:detailed
```

#### `/summary_help`
Display help information about all commands.

## Project Structure

```
nlp-llmAgent-miawDuwa/
├── bot.py                    # Main bot application
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (create from .env.example)
├── .env.example             # Example environment file
├── .gitignore               # Git ignore rules
├── README.md                # This file
├── database/
│   └── messages.db          # SQLite database (auto-created)
└── utils/
    ├── __init__.py          # Package initializer
    ├── message_tracker.py   # Message storage and retrieval
    ├── llm_handler.py       # OpenAI API integration
    └── summarizer.py        # Summarization orchestration
```

## How It Works

1. **Message Tracking**: The bot listens to all messages in channels it has access to and stores them in a SQLite database.

2. **Activity Tracking**: When users send messages, their "last seen" timestamp is updated for that channel.

3. **Summarization Process**:
   - User requests a summary via slash command
   - Bot retrieves relevant messages from database
   - Messages are formatted and sent to OpenAI API
   - GPT model generates a structured summary
   - Summary is sent back to the user

4. **Cleanup**: Old messages are automatically cleaned up every 24 hours based on `MESSAGE_RETENTION_DAYS` setting.

## Configuration

Edit values in `.env`:

- `MAX_MESSAGES_TO_SUMMARIZE`: Maximum number of messages to include in a summary (default: 200)
- `MESSAGE_RETENTION_DAYS`: How many days to keep messages in database (default: 7)
- `SUMMARY_MAX_LENGTH`: Maximum characters for summary response (default: 2000)

You can also modify the LLM model in `utils/llm_handler.py`:
```python
self.model = "gpt-4o-mini"  # Change to "gpt-4o" for better quality (higher cost)
```

## Cost Considerations

Using `gpt-4o-mini` (default):
- ~$0.150 per 1M input tokens
- ~$0.600 per 1M output tokens
- A typical 100-message summary costs approximately $0.01-0.03

For reference:
- 100 messages ≈ 5,000-10,000 tokens
- Summary response ≈ 500-1,000 tokens

## Troubleshooting

### Bot doesn't respond to commands
- Ensure Message Content Intent is enabled in Discord Developer Portal
- Check that bot has necessary permissions in your server
- Run `/summary_help` to verify slash commands are registered

### "Invalid time format" error
- Use format: number + unit (e.g., `2h`, `30m`, `1d`)
- Valid units: `m` (minutes), `h` (hours), `d` (days)

### Database errors
- Ensure `database/` directory exists
- Check file permissions for writing to database

### OpenAI API errors
- Verify your API key is correct in `.env`
- Check your OpenAI account has available credits
- Review OpenAI API status at status.openai.com

## Future Enhancements

Possible improvements:
- Web dashboard for analytics
- Sentiment analysis of conversations
- Thread-aware summarization
- Multi-language support
- Summary scheduling (daily digest)
- Export summaries to email or Notion
- Voice channel activity summaries

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is open source and available under the MIT License.
