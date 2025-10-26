import discord
from discord import app_commands
from discord.ext import commands, tasks
import os
from dotenv import load_dotenv
from utils import MessageTracker, LLMHandler, Summarizer
import asyncio

# Load environment variables
load_dotenv()

# Configuration
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MAX_MESSAGES = int(os.getenv('MAX_MESSAGES_TO_SUMMARIZE', 200))
MESSAGE_RETENTION_DAYS = int(os.getenv('MESSAGE_RETENTION_DAYS', 7))

# Bot setup with required intents
intents = discord.Intents.default()
intents.message_content = True  # Required to read message content
intents.messages = True
intents.guilds = True
intents.members = True

class SummaryBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix='!', intents=intents)
        self.message_tracker = MessageTracker()
        self.llm_handler = LLMHandler(OPENAI_API_KEY)
        self.summarizer = Summarizer(self.message_tracker, self.llm_handler)

    async def setup_hook(self):
        """Called when the bot is starting up"""
        await self.message_tracker.initialize()
        print("Database initialized")

        # Start background tasks
        self.cleanup_messages.start()

        # Sync slash commands
        await self.tree.sync()
        print("Slash commands synced")

    async def on_ready(self):
        """Called when bot is ready"""
        print(f'{self.user} has connected to Discord!')
        print(f'Bot is in {len(self.guilds)} guilds')

    async def on_message(self, message: discord.Message):
        """Track all messages"""
        # Ignore bot messages
        if message.author.bot:
            return

        # Only track messages in guilds (not DMs)
        if message.guild:
            await self.message_tracker.store_message(message)

            # Update user activity
            await self.message_tracker.update_user_activity(
                message.author.id,
                message.guild.id,
                message.channel.id
            )

        # Process commands
        await self.process_commands(message)

    @tasks.loop(hours=24)
    async def cleanup_messages(self):
        """Clean up old messages daily"""
        await self.message_tracker.cleanup_old_messages(MESSAGE_RETENTION_DAYS)
        print(f"Cleaned up messages older than {MESSAGE_RETENTION_DAYS} days")

    @cleanup_messages.before_loop
    async def before_cleanup(self):
        """Wait for bot to be ready before starting cleanup task"""
        await self.wait_until_ready()


# Create bot instance
bot = SummaryBot()


# Slash Commands
@bot.tree.command(name="summarize", description="Summarize chat messages")
@app_commands.describe(
    mode="Summarization mode: 'catchup' (since last seen), time-based (e.g., '2h'), or context-based",
    context="Optional: Topic to focus on (for context-based summaries)",
    detail="Summary detail level: brief, general, or detailed",
    language="Summary language (default: English)"
)
@app_commands.choices(
    detail=[
        app_commands.Choice(name="Brief", value="brief"),
        app_commands.Choice(name="General", value="general"),
        app_commands.Choice(name="Detailed", value="detailed")
    ],
    language=[
        app_commands.Choice(name="English", value="english"),
        app_commands.Choice(name="Indonesian", value="indonesian")
    ]
)
async def summarize(
    interaction: discord.Interaction,
    mode: str = "catchup",
    context: str = None,
    detail: app_commands.Choice[str] = None,
    language: app_commands.Choice[str] = None
):
    """Main summarize command"""
    await interaction.response.defer(thinking=True)

    summary_type = detail.value if detail else "general"
    lang = language.value if language else "english"
    channel_id = interaction.channel.id
    user_id = interaction.user.id
    guild_id = interaction.guild.id

    try:
        # Determine summarization type based on mode
        if mode == "catchup" or mode == "catch-up":
            summary = await bot.summarizer.summarize_since_last_seen(
                user_id, guild_id, channel_id, summary_type, MAX_MESSAGES, lang
            )

        elif context:
            # Context-based summary
            hours_back = 24
            # Check if mode is a time string
            if bot.summarizer.parse_time_string(mode):
                time_delta = bot.summarizer.parse_time_string(mode)
                hours_back = int(time_delta.total_seconds() / 3600)

            summary = await bot.summarizer.summarize_with_context(
                channel_id, context, hours_back, MAX_MESSAGES, lang
            )

        else:
            # Time-based summary (e.g., "2h", "1d")
            summary = await bot.summarizer.summarize_time_range(
                channel_id, mode, summary_type, MAX_MESSAGES, lang
            )

        # Split long summaries if needed (Discord has 2000 char limit)
        if len(summary) > 2000:
            chunks = [summary[i:i+2000] for i in range(0, len(summary), 2000)]
            await interaction.followup.send(chunks[0])
            for chunk in chunks[1:]:
                await interaction.channel.send(chunk)
        else:
            await interaction.followup.send(summary)

    except Exception as e:
        await interaction.followup.send(f"Error generating summary: {str(e)}")
        print(f"Error in summarize command: {e}")


@bot.tree.command(name="catchup", description="Quick bullet-point catchup of missed messages")
@app_commands.describe(
    language="Summary language (default: English)"
)
@app_commands.choices(language=[
    app_commands.Choice(name="English", value="english"),
    app_commands.Choice(name="Indonesian", value="indonesian")
])
async def catchup(interaction: discord.Interaction, language: app_commands.Choice[str] = None):
    """Quick catchup command"""
    await interaction.response.defer(thinking=True)

    lang = language.value if language else "english"

    try:
        summary = await bot.summarizer.quick_catchup(
            interaction.user.id,
            interaction.guild.id,
            interaction.channel.id,
            language=lang
        )

        await interaction.followup.send(summary)

    except Exception as e:
        await interaction.followup.send(f"Error generating catchup: {str(e)}")
        print(f"Error in catchup command: {e}")


@bot.tree.command(name="summary_help", description="Show help for summary commands")
async def summary_help(interaction: discord.Interaction):
    """Help command"""
    help_text = """
**Chat Summarization Bot Help**

**Commands:**

`/catchup` - Quick bullet-point summary of messages since you were last active
  • `language`: Choose English or Indonesian (default: English)

`/summarize` - Summarize messages with options:
  • `mode`: How to summarize
    - `catchup` - Since you were last active (default)
    - `2h`, `30m`, `1d` - Last X hours/minutes/days
    - Any text + context parameter - Context-based summary

  • `context`: Optional topic to focus on (e.g., "meeting times", "project updates")

  • `detail`: Summary detail level
    - `brief` - 2-3 sentence overview
    - `general` - Balanced summary (default)
    - `detailed` - Comprehensive summary with topics

  • `language`: Choose English or Indonesian (default: English)

**Examples:**
`/catchup` - Quick catchup since last seen (English)
`/catchup language:Indonesian` - Quick catchup in Indonesian
`/summarize mode:2h` - Summarize last 2 hours
`/summarize mode:1d detail:detailed` - Detailed summary of last 24h
`/summarize mode:2h language:Indonesian` - Summarize in Indonesian
`/summarize mode:12h context:meeting` - Summarize meeting-related messages from last 12h
`/summarize mode:1d context:project language:Indonesian` - Context summary in Indonesian

**Note:** The bot tracks messages automatically. Your "last seen" time updates when you send messages.
"""

    await interaction.response.send_message(help_text, ephemeral=True)


# Traditional prefix command for testing
@bot.command(name='ping')
async def ping(ctx):
    """Test command"""
    await ctx.send(f'Pong! Latency: {round(bot.latency * 1000)}ms')


# Error handling
@bot.event
async def on_command_error(ctx, error):
    """Handle command errors"""
    if isinstance(error, commands.CommandNotFound):
        return
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"Missing required argument: {error.param}")
    else:
        await ctx.send(f"An error occurred: {str(error)}")
        print(f"Error: {error}")


@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    """Handle slash command errors"""
    if isinstance(error, app_commands.CommandOnCooldown):
        await interaction.response.send_message(
            f"This command is on cooldown. Try again in {error.retry_after:.2f}s",
            ephemeral=True
        )
    else:
        await interaction.response.send_message(
            f"An error occurred: {str(error)}",
            ephemeral=True
        )
        print(f"Slash command error: {error}")


# Run the bot
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("Error: DISCORD_TOKEN not found in environment variables")
        print("Please create a .env file with your Discord bot token")
        exit(1)

    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please add your OpenAI API key to the .env file")
        exit(1)

    bot.run(DISCORD_TOKEN)
