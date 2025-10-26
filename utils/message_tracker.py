import aiosqlite
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class MessageTracker:
    """Handles message storage and retrieval from SQLite database"""

    def __init__(self, db_path: str = "database/messages.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    async def initialize(self):
        """Initialize database tables"""
        async with aiosqlite.connect(self.db_path) as db:
            # Messages table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY,
                    channel_id INTEGER NOT NULL,
                    guild_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    username TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    has_attachments BOOLEAN DEFAULT 0,
                    reply_to INTEGER
                )
            """)

            # User activity tracking
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_activity (
                    user_id INTEGER,
                    guild_id INTEGER,
                    channel_id INTEGER,
                    last_seen TEXT NOT NULL,
                    PRIMARY KEY (user_id, guild_id, channel_id)
                )
            """)

            # Create indexes for faster queries
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_channel_timestamp
                ON messages(channel_id, timestamp)
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_guild_timestamp
                ON messages(guild_id, timestamp)
            """)

            await db.commit()

    async def store_message(self, message):
        """Store a Discord message"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO messages
                (message_id, channel_id, guild_id, user_id, username, content, timestamp, has_attachments, reply_to)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message.id,
                message.channel.id,
                message.guild.id if message.guild else 0,
                message.author.id,
                str(message.author),
                message.content,
                message.created_at.isoformat(),
                len(message.attachments) > 0,
                message.reference.message_id if message.reference else None
            ))
            await db.commit()

    async def update_user_activity(self, user_id: int, guild_id: int, channel_id: int):
        """Update user's last seen timestamp"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO user_activity
                (user_id, guild_id, channel_id, last_seen)
                VALUES (?, ?, ?, ?)
            """, (user_id, guild_id, channel_id, datetime.utcnow().isoformat()))
            await db.commit()

    async def get_user_last_seen(self, user_id: int, guild_id: int, channel_id: int) -> Optional[datetime]:
        """Get user's last seen timestamp for a channel"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT last_seen FROM user_activity
                WHERE user_id = ? AND guild_id = ? AND channel_id = ?
            """, (user_id, guild_id, channel_id))
            row = await cursor.fetchone()

            if row:
                return datetime.fromisoformat(row[0])
            return None

    async def get_messages_since(
        self,
        channel_id: int,
        since: datetime,
        limit: int = 200
    ) -> List[Dict]:
        """Get messages from a channel since a specific time"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT message_id, user_id, username, content, timestamp, has_attachments, reply_to
                FROM messages
                WHERE channel_id = ? AND timestamp > ?
                ORDER BY timestamp ASC
                LIMIT ?
            """, (channel_id, since.isoformat(), limit))

            rows = await cursor.fetchall()

            return [
                {
                    "message_id": row[0],
                    "user_id": row[1],
                    "username": row[2],
                    "content": row[3],
                    "timestamp": row[4],
                    "has_attachments": bool(row[5]),
                    "reply_to": row[6]
                }
                for row in rows
            ]

    async def get_messages_in_timerange(
        self,
        channel_id: int,
        start: datetime,
        end: datetime,
        limit: int = 200
    ) -> List[Dict]:
        """Get messages from a channel within a time range"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT message_id, user_id, username, content, timestamp, has_attachments, reply_to
                FROM messages
                WHERE channel_id = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
                LIMIT ?
            """, (channel_id, start.isoformat(), end.isoformat(), limit))

            rows = await cursor.fetchall()

            return [
                {
                    "message_id": row[0],
                    "user_id": row[1],
                    "username": row[2],
                    "content": row[3],
                    "timestamp": row[4],
                    "has_attachments": bool(row[5]),
                    "reply_to": row[6]
                }
                for row in rows
            ]

    async def cleanup_old_messages(self, days: int = 7):
        """Remove messages older than specified days"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                DELETE FROM messages WHERE timestamp < ?
            """, (cutoff.isoformat(),))
            await db.commit()
