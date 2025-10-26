from datetime import datetime, timedelta
from typing import Optional, List, Dict
import re
from .message_tracker import MessageTracker
from .llm_handler import LLMHandler

class Summarizer:
    """Orchestrates message retrieval and summarization"""

    def __init__(self, message_tracker: MessageTracker, llm_handler: LLMHandler):
        self.message_tracker = message_tracker
        self.llm_handler = llm_handler

    def parse_time_string(self, time_str: str) -> Optional[timedelta]:
        """Parse time strings like '2h', '30m', '1d' into timedelta"""
        pattern = r'^(\d+)([mhd])$'
        match = re.match(pattern, time_str.lower())

        if not match:
            return None

        value = int(match.group(1))
        unit = match.group(2)

        if unit == 'm':
            return timedelta(minutes=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'd':
            return timedelta(days=value)

        return None

    async def summarize_since_last_seen(
        self,
        user_id: int,
        guild_id: int,
        channel_id: int,
        summary_type: str = "general",
        max_messages: int = 200,
        language: str = "english"
    ) -> str:
        """Summarize messages since user was last active"""

        last_seen = await self.message_tracker.get_user_last_seen(user_id, guild_id, channel_id)

        if not last_seen:
            if language.lower() in ["indonesian", "indonesia"]:
                return "Saya tidak memiliki catatan kapan Anda terakhir aktif di channel ini. Coba gunakan ringkasan berbasis waktu (misalnya, `/summarize 2h`)."
            return "I don't have a record of when you were last active in this channel. Try using a time-based summary instead (e.g., `/summarize 2h`)."

        messages = await self.message_tracker.get_messages_since(
            channel_id,
            last_seen,
            limit=max_messages
        )

        if not messages:
            time_ago = self._format_time_ago(last_seen, language)
            if language.lower() in ["indonesian", "indonesia"]:
                return f"Tidak ada pesan baru sejak Anda terakhir aktif ({time_ago})."
            return f"No new messages since you were last active ({time_ago})."

        # Update last seen to now
        await self.message_tracker.update_user_activity(user_id, guild_id, channel_id)

        return await self.llm_handler.generate_summary(messages, summary_type=summary_type, language=language)

    async def summarize_time_range(
        self,
        channel_id: int,
        time_str: str,
        summary_type: str = "general",
        max_messages: int = 200,
        language: str = "english"
    ) -> str:
        """Summarize messages from the last X hours/days"""

        time_delta = self.parse_time_string(time_str)

        if not time_delta:
            if language.lower() in ["indonesian", "indonesia"]:
                return "Format waktu tidak valid. Gunakan format seperti: `2h` (2 jam), `30m` (30 menit), atau `1d` (1 hari)"
            return "Invalid time format. Use format like: `2h` (2 hours), `30m` (30 minutes), or `1d` (1 day)"

        end_time = datetime.utcnow()
        start_time = end_time - time_delta

        messages = await self.message_tracker.get_messages_in_timerange(
            channel_id,
            start_time,
            end_time,
            limit=max_messages
        )

        if not messages:
            if language.lower() in ["indonesian", "indonesia"]:
                return f"Tidak ada pesan yang ditemukan dalam {time_str} terakhir."
            return f"No messages found in the last {time_str}."

        return await self.llm_handler.generate_summary(messages, summary_type=summary_type, language=language)

    async def summarize_with_context(
        self,
        channel_id: int,
        context: str,
        hours_back: int = 24,
        max_messages: int = 200,
        language: str = "english"
    ) -> str:
        """Summarize messages with a specific topic/context filter"""

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)

        messages = await self.message_tracker.get_messages_in_timerange(
            channel_id,
            start_time,
            end_time,
            limit=max_messages
        )

        if not messages:
            if language.lower() in ["indonesian", "indonesia"]:
                return f"Tidak ada pesan yang ditemukan dalam {hours_back} jam terakhir."
            return f"No messages found in the last {hours_back} hours."

        # Filter messages that might be relevant to context
        # (basic keyword matching, LLM will do deeper filtering)
        context_lower = context.lower()
        relevant_messages = [
            msg for msg in messages
            if context_lower in msg['content'].lower()
        ]

        # If filtering is too aggressive, use all messages and let LLM filter
        messages_to_summarize = relevant_messages if len(relevant_messages) >= 5 else messages

        return await self.llm_handler.generate_summary(
            messages_to_summarize,
            context=context,
            summary_type="context",
            language=language
        )

    async def quick_catchup(
        self,
        user_id: int,
        guild_id: int,
        channel_id: int,
        max_messages: int = 50,
        language: str = "english"
    ) -> str:
        """Quick bullet-point catchup since last seen"""

        last_seen = await self.message_tracker.get_user_last_seen(user_id, guild_id, channel_id)

        if not last_seen:
            # Default to last 2 hours if no record
            last_seen = datetime.utcnow() - timedelta(hours=2)

        messages = await self.message_tracker.get_messages_since(
            channel_id,
            last_seen,
            limit=max_messages
        )

        if not messages:
            if language.lower() in ["indonesian", "indonesia"]:
                return "Tidak ada pesan baru untuk diikuti!"
            return "No new messages to catch up on!"

        # Update last seen
        await self.message_tracker.update_user_activity(user_id, guild_id, channel_id)

        return await self.llm_handler.generate_quick_catchup(messages, language=language)

    def _format_time_ago(self, dt: datetime, language: str = "english") -> str:
        """Format a datetime as 'X time ago'"""
        delta = datetime.utcnow() - dt

        if language.lower() in ["indonesian", "indonesia"]:
            if delta.days > 0:
                return f"{delta.days} hari yang lalu"
            elif delta.seconds >= 3600:
                hours = delta.seconds // 3600
                return f"{hours} jam yang lalu"
            elif delta.seconds >= 60:
                minutes = delta.seconds // 60
                return f"{minutes} menit yang lalu"
            else:
                return "baru saja"
        else:
            if delta.days > 0:
                return f"{delta.days} day{'s' if delta.days != 1 else ''} ago"
            elif delta.seconds >= 3600:
                hours = delta.seconds // 3600
                return f"{hours} hour{'s' if hours != 1 else ''} ago"
            elif delta.seconds >= 60:
                minutes = delta.seconds // 60
                return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
            else:
                return "just now"
