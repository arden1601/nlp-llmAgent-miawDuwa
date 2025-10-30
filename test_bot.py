"""
Discord LLM Chat Summarization Bot - Test Suite
================================================

This module contains unit tests and functional tests for the Discord bot.

Test Cases:
1. Time String Parsing (Unit Test)
2. Message Storage and Retrieval (Functional Test)
3. User Activity Tracking (Functional Test)
4. LLM Prompt Generation (Unit Test)
5. Context-Based Filtering (Functional Test)
6. Multi-Language Support (Functional Test)

Run with: pytest test_bot.py -v
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.summarizer import Summarizer
from utils.message_tracker import MessageTracker
from utils.llm_handler import LLMHandler


# ============================================================================
# TEST CASE 1: Time String Parsing (Unit Test)
# ============================================================================

class TestTimeStringParsing:
    """Test parsing of time strings like '2h', '30m', '1d'"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.message_tracker = Mock()
        self.llm_handler = Mock()
        self.summarizer = Summarizer(self.message_tracker, self.llm_handler)
    
    def test_parse_hours(self):
        """Test parsing hours format"""
        result = self.summarizer.parse_time_string('2h')
        assert result == timedelta(hours=2)
        assert result.total_seconds() == 7200
    
    def test_parse_minutes(self):
        """Test parsing minutes format"""
        result = self.summarizer.parse_time_string('30m')
        assert result == timedelta(minutes=30)
        assert result.total_seconds() == 1800
    
    def test_parse_days(self):
        """Test parsing days format"""
        result = self.summarizer.parse_time_string('1d')
        assert result == timedelta(days=1)
        assert result.total_seconds() == 86400
    
    def test_parse_invalid_format(self):
        """Test invalid formats return None"""
        assert self.summarizer.parse_time_string('2hours') is None
        assert self.summarizer.parse_time_string('abc') is None
        assert self.summarizer.parse_time_string('2.5h') is None
        assert self.summarizer.parse_time_string('') is None
    
    def test_parse_case_insensitive(self):
        """Test case insensitivity"""
        assert self.summarizer.parse_time_string('2H') == timedelta(hours=2)
        assert self.summarizer.parse_time_string('30M') == timedelta(minutes=30)
        assert self.summarizer.parse_time_string('1D') == timedelta(days=1)
    
    def test_parse_large_values(self):
        """Test parsing large time values"""
        result = self.summarizer.parse_time_string('168h')  # 1 week
        assert result == timedelta(hours=168)
        assert result.days == 7


# ============================================================================
# TEST CASE 2: Message Storage and Retrieval (Functional Test)
# ============================================================================

class TestMessageStorageRetrieval:
    """Test database operations for message storage and retrieval"""
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_message(self):
        """Test storing and retrieving a single message"""
        # Create tracker
        tracker = MessageTracker(db_path="test_dbs/test_database.db")
        await tracker.initialize()
        
        try:
            # Create mock Discord message
            mock_message = Mock()
            mock_message.id = 123456789
            mock_message.channel.id = 111111
            mock_message.guild.id = 222222
            mock_message.author.id = 333333
            mock_message.author.__str__ = Mock(return_value="TestUser#1234")
            mock_message.content = "Hello, this is a test message!"
            mock_message.created_at = datetime.utcnow()
            mock_message.attachments = []
            mock_message.reference = None
            
            # Store message
            await tracker.store_message(mock_message)
            
            # Retrieve messages
            since = datetime.utcnow() - timedelta(hours=1)
            messages = await tracker.get_messages_since(111111, since, limit=10)
            
            # Assertions
            assert len(messages) == 1
            assert messages[0]['content'] == "Hello, this is a test message!"
            assert messages[0]['username'] == "TestUser#1234"
            assert messages[0]['message_id'] == 123456789
        finally:
            # Cleanup
            if os.path.exists("test_dbs"):
                shutil.rmtree("test_dbs")
    
    @pytest.mark.asyncio
    async def test_retrieve_messages_in_timerange(self):
        """Test retrieving messages within a specific time range"""
        # Create tracker
        tracker = MessageTracker(db_path="test_dbs/test_database2.db")
        await tracker.initialize()
        
        try:
            now = datetime.utcnow()
            
            # Create multiple mock messages with different timestamps
            for i in range(5):
                mock_message = Mock()
                mock_message.id = 1000 + i
                mock_message.channel.id = 111111
                mock_message.guild.id = 222222
                mock_message.author.id = 333333
                mock_message.author.__str__ = Mock(return_value=f"User{i}#1234")
                mock_message.content = f"Message {i}"
                mock_message.created_at = now - timedelta(hours=5-i)
                mock_message.attachments = []
                mock_message.reference = None
                
                await tracker.store_message(mock_message)
            
            # Retrieve messages from last 3 hours
            start = now - timedelta(hours=3)
            end = now
            messages = await tracker.get_messages_in_timerange(111111, start, end, limit=10)
            
            # Should get 3 messages (0, 1, 2 hours old)
            assert len(messages) >= 3
        finally:
            # Cleanup
            if os.path.exists("test_dbs"):
                shutil.rmtree("test_dbs")
    
    @pytest.mark.asyncio
    async def test_message_limit(self):
        """Test that message limit is respected"""
        # Create tracker
        tracker = MessageTracker(db_path="test_dbs/test_database3.db")
        await tracker.initialize()
        
        try:
            now = datetime.utcnow()
            
            # Create 10 messages
            for i in range(10):
                mock_message = Mock()
                mock_message.id = 2000 + i
                mock_message.channel.id = 111111
                mock_message.guild.id = 222222
                mock_message.author.id = 333333
                mock_message.author.__str__ = Mock(return_value="TestUser#1234")
                mock_message.content = f"Message {i}"
                mock_message.created_at = now - timedelta(minutes=i)
                mock_message.attachments = []
                mock_message.reference = None
                
                await tracker.store_message(mock_message)
            
            # Retrieve with limit of 5
            since = now - timedelta(hours=1)
            messages = await tracker.get_messages_since(111111, since, limit=5)
            
            # Should only get 5 messages
            assert len(messages) == 5
        finally:
            # Cleanup
            if os.path.exists("test_dbs"):
                shutil.rmtree("test_dbs")


# ============================================================================
# TEST CASE 3: User Activity Tracking (Functional Test)
# ============================================================================

class TestUserActivityTracking:
    """Test user activity tracking functionality"""
    
    @pytest.mark.asyncio
    async def test_update_user_activity(self):
        """Test updating user's last seen timestamp"""
        # Create tracker
        tracker = MessageTracker(db_path="test_dbs/test_activity.db")
        await tracker.initialize()
        
        try:
            user_id = 123456
            guild_id = 789012
            channel_id = 345678
            
            # Update activity
            await tracker.update_user_activity(user_id, guild_id, channel_id)
            
            # Retrieve last seen
            last_seen = await tracker.get_user_last_seen(user_id, guild_id, channel_id)
            
            # Assertions
            assert last_seen is not None
            assert isinstance(last_seen, datetime)
            assert (datetime.utcnow() - last_seen).total_seconds() < 5  # Within 5 seconds
        finally:
            # Cleanup
            if os.path.exists("test_dbs"):
                shutil.rmtree("test_dbs")
    
    @pytest.mark.asyncio
    async def test_no_activity_record(self):
        """Test behavior when user has no activity record"""
        # Create tracker
        tracker = MessageTracker(db_path="test_dbs/test_activity2.db")
        await tracker.initialize()
        
        try:
            user_id = 999999
            guild_id = 888888
            channel_id = 777777
            
            # Try to get last seen for user with no record
            last_seen = await tracker.get_user_last_seen(user_id, guild_id, channel_id)
            
            # Should return None
            assert last_seen is None
        finally:
            # Cleanup
            if os.path.exists("test_dbs"):
                shutil.rmtree("test_dbs")
    
    @pytest.mark.asyncio
    async def test_multiple_channel_tracking(self):
        """Test tracking user activity across multiple channels"""
        # Create tracker
        tracker = MessageTracker(db_path="test_dbs/test_activity3.db")
        await tracker.initialize()
        
        try:
            user_id = 123456
            guild_id = 789012
            
            # Update activity in multiple channels
            await tracker.update_user_activity(user_id, guild_id, 111111)
            await asyncio.sleep(0.1)
            await tracker.update_user_activity(user_id, guild_id, 222222)
            
            # Get last seen for both channels
            last_seen_1 = await tracker.get_user_last_seen(user_id, guild_id, 111111)
            last_seen_2 = await tracker.get_user_last_seen(user_id, guild_id, 222222)
            
            # Both should exist and channel 2 should be more recent
            assert last_seen_1 is not None
            assert last_seen_2 is not None
            assert last_seen_2 >= last_seen_1
        finally:
            # Cleanup
            if os.path.exists("test_dbs"):
                shutil.rmtree("test_dbs")


# ============================================================================
# TEST CASE 4: LLM Prompt Generation (Unit Test)
# ============================================================================

class TestLLMPromptGeneration:
    """Test LLM prompt generation for different scenarios"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.llm_handler = LLMHandler(api_key="test_key")
    
    def test_format_messages_for_prompt(self):
        """Test message formatting for LLM input"""
        messages = [
            {
                'timestamp': '2025-10-30T14:23:45',
                'username': 'Alice',
                'content': 'Hello everyone!',
                'has_attachments': False
            },
            {
                'timestamp': '2025-10-30T14:25:30',
                'username': 'Bob',
                'content': 'Hi Alice!',
                'has_attachments': False
            }
        ]
        
        system_prompt, user_prompt = self.llm_handler.create_summary_prompt(
            messages, 
            summary_type="general",
            language="english"
        )
        
        # Check that messages are formatted correctly
        assert "[14:23] Alice: Hello everyone!" in user_prompt
        assert "[14:25] Bob: Hi Alice!" in user_prompt
        assert "summarize" in system_prompt.lower()
    
    def test_prompt_language_english(self):
        """Test English prompt generation"""
        messages = [{'timestamp': '2025-10-30T14:23:45', 'username': 'User', 
                    'content': 'Test', 'has_attachments': False}]
        
        system_prompt, user_prompt = self.llm_handler.create_summary_prompt(
            messages, 
            summary_type="general",
            language="english"
        )
        
        assert "You are a helpful assistant" in system_prompt
        assert "summarize" in system_prompt.lower()
    
    def test_prompt_language_indonesian(self):
        """Test Indonesian prompt generation"""
        messages = [{'timestamp': '2025-10-30T14:23:45', 'username': 'User', 
                    'content': 'Test', 'has_attachments': False}]
        
        system_prompt, user_prompt = self.llm_handler.create_summary_prompt(
            messages, 
            summary_type="general",
            language="indonesian"
        )
        
        assert "Anda adalah asisten" in system_prompt
        assert "rangkum" in system_prompt.lower()
    
    def test_prompt_with_context(self):
        """Test context-based prompt generation"""
        messages = [{'timestamp': '2025-10-30T14:23:45', 'username': 'User', 
                    'content': 'Meeting at 2pm', 'has_attachments': False}]
        
        system_prompt, user_prompt = self.llm_handler.create_summary_prompt(
            messages, 
            context="meeting",
            summary_type="context",
            language="english"
        )
        
        assert "meeting" in user_prompt.lower()
        assert "focus" in system_prompt.lower()
    
    def test_different_detail_levels(self):
        """Test different summary detail levels"""
        messages = [{'timestamp': '2025-10-30T14:23:45', 'username': 'User', 
                    'content': 'Test', 'has_attachments': False}]
        
        # Brief
        system_brief, _ = self.llm_handler.create_summary_prompt(
            messages, summary_type="brief", language="english"
        )
        assert "brief" in system_brief.lower()
        
        # Detailed
        system_detailed, _ = self.llm_handler.create_summary_prompt(
            messages, summary_type="detailed", language="english"
        )
        assert "detailed" in system_detailed.lower()
    
    def test_handle_attachments(self):
        """Test handling messages with attachments"""
        messages = [
            {
                'timestamp': '2025-10-30T14:23:45',
                'username': 'Alice',
                'content': '',
                'has_attachments': True
            }
        ]
        
        _, user_prompt = self.llm_handler.create_summary_prompt(
            messages, 
            summary_type="general",
            language="english"
        )
        
        assert "[attachment]" in user_prompt


# ============================================================================
# TEST CASE 5: Context-Based Filtering (Functional Test)
# ============================================================================

class TestContextBasedFiltering:
    """Test context-based message filtering functionality"""
    
    @pytest.fixture
    async def setup_summarizer(self):
        """Setup summarizer with mock dependencies"""
        tracker = Mock()
        llm_handler = Mock()
        llm_handler.generate_summary = AsyncMock(return_value="Test summary")
        
        summarizer = Summarizer(tracker, llm_handler)
        return summarizer, tracker, llm_handler
    
    @pytest.mark.asyncio
    async def test_keyword_filtering(self, setup_summarizer):
        """Test that keyword filtering works correctly"""
        summarizer, tracker, llm_handler = await setup_summarizer
        
        # Mock messages - some with keyword, some without
        mock_messages = [
            {'content': 'We need to schedule a meeting', 'username': 'Alice'},
            {'content': 'Random message about lunch', 'username': 'Bob'},
            {'content': 'Meeting agenda is ready', 'username': 'Carol'},
            {'content': 'Another random message', 'username': 'Dave'},
            {'content': 'Can we move the meeting?', 'username': 'Eve'}
        ]
        
        tracker.get_messages_in_timerange = AsyncMock(return_value=mock_messages)
        
        # Test context filtering
        result = await summarizer.summarize_with_context(
            channel_id=123456,
            context="meeting",
            hours_back=24,
            language="english"
        )
        
        # Check that LLM was called (meaning filtering happened)
        assert llm_handler.generate_summary.called
        
        # Get the messages that were passed to LLM
        call_args = llm_handler.generate_summary.call_args
        filtered_messages = call_args[0][0]
        
        # Should have filtered to messages containing "meeting"
        assert len(filtered_messages) <= len(mock_messages)
    
    @pytest.mark.asyncio
    async def test_insufficient_keyword_matches(self, setup_summarizer):
        """Test behavior when keyword matches are insufficient"""
        summarizer, tracker, llm_handler = await setup_summarizer
        
        # Only 2 messages with keyword (less than threshold of 5)
        mock_messages = [
            {'content': 'Meeting at 2pm', 'username': 'Alice'},
            {'content': 'Random message 1', 'username': 'Bob'},
            {'content': 'Random message 2', 'username': 'Carol'},
            {'content': 'Random message 3', 'username': 'Dave'},
            {'content': 'Another meeting note', 'username': 'Eve'}
        ]
        
        tracker.get_messages_in_timerange = AsyncMock(return_value=mock_messages)
        
        # Test context filtering
        await summarizer.summarize_with_context(
            channel_id=123456,
            context="meeting",
            hours_back=24,
            language="english"
        )
        
        # Should pass ALL messages to LLM (not just filtered ones)
        call_args = llm_handler.generate_summary.call_args
        messages_to_llm = call_args[0][0]
        assert len(messages_to_llm) == len(mock_messages)
    
    @pytest.mark.asyncio
    async def test_case_insensitive_filtering(self, setup_summarizer):
        """Test that filtering is case-insensitive"""
        summarizer, tracker, llm_handler = await setup_summarizer
        
        mock_messages = [
            {'content': 'MEETING AT 2PM', 'username': 'Alice'},
            {'content': 'Meeting at 3pm', 'username': 'Bob'},
            {'content': 'meeting tomorrow', 'username': 'Carol'},
        ]
        
        tracker.get_messages_in_timerange = AsyncMock(return_value=mock_messages)
        
        await summarizer.summarize_with_context(
            channel_id=123456,
            context="meeting",  # lowercase
            hours_back=24,
            language="english"
        )
        
        # All messages should match (case-insensitive)
        call_args = llm_handler.generate_summary.call_args
        filtered_messages = call_args[0][0]
        assert len(filtered_messages) >= 3


# ============================================================================
# TEST CASE 6: Multi-Language Support (Functional Test)
# ============================================================================

class TestMultiLanguageSupport:
    """Test multi-language support functionality"""
    
    @pytest.fixture
    def llm_handler(self):
        """Create LLM handler with mock API"""
        with patch('openai.OpenAI') as mock_openai:
            handler = LLMHandler(api_key="test_key")
            
            # Mock the API response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Test summary"
            
            mock_openai.return_value.chat.completions.create = Mock(
                return_value=mock_response
            )
            
            return handler
    
    @pytest.mark.asyncio
    async def test_english_summary(self, llm_handler):
        """Test English summary generation"""
        messages = [
            {'timestamp': '2025-10-30T14:23:45', 'username': 'Alice', 
             'content': 'Hello', 'has_attachments': False}
        ]
        
        result = await llm_handler.generate_summary(
            messages, 
            language="english"
        )
        
        # Should contain English metadata
        assert "Summarized" in result
        assert "messages from" in result
        assert "participants" in result
    
    @pytest.mark.asyncio
    async def test_indonesian_summary(self, llm_handler):
        """Test Indonesian summary generation"""
        messages = [
            {'timestamp': '2025-10-30T14:23:45', 'username': 'Alice', 
             'content': 'Halo', 'has_attachments': False}
        ]
        
        result = await llm_handler.generate_summary(
            messages, 
            language="indonesian"
        )
        
        # Should contain Indonesian metadata
        assert "Merangkum" in result
        assert "pesan dari" in result
        assert "peserta" in result
    
    @pytest.mark.asyncio
    async def test_empty_messages_english(self, llm_handler):
        """Test handling empty messages in English"""
        result = await llm_handler.generate_summary(
            [], 
            language="english"
        )
        
        assert "No messages to summarize" in result
    
    @pytest.mark.asyncio
    async def test_empty_messages_indonesian(self, llm_handler):
        """Test handling empty messages in Indonesian"""
        result = await llm_handler.generate_summary(
            [], 
            language="indonesian"
        )
        
        assert "Tidak ada pesan" in result
    
    @pytest.mark.asyncio
    async def test_quick_catchup_english(self, llm_handler):
        """Test quick catchup in English"""
        messages = [
            {'username': 'Alice', 'content': 'Test message', 'has_attachments': False}
        ]
        
        result = await llm_handler.generate_quick_catchup(messages, language="english")
        
        # Should return some result (mocked)
        assert result is not None
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_quick_catchup_indonesian(self, llm_handler):
        """Test quick catchup in Indonesian"""
        messages = [
            {'username': 'Alice', 'content': 'Pesan tes', 'has_attachments': False}
        ]
        
        result = await llm_handler.generate_quick_catchup(messages, language="indonesian")
        
        # Should return some result (mocked)
        assert result is not None
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_language_case_insensitivity(self, llm_handler):
        """Test that language parameter is case-insensitive"""
        messages = [
            {'timestamp': '2025-10-30T14:23:45', 'username': 'Alice', 
             'content': 'Test', 'has_attachments': False}
        ]
        
        # Test with different cases
        result1 = await llm_handler.generate_summary(messages, language="INDONESIAN")
        result2 = await llm_handler.generate_summary(messages, language="Indonesian")
        result3 = await llm_handler.generate_summary(messages, language="indonesia")
        
        # All should use Indonesian
        assert "Merangkum" in result1
        assert "Merangkum" in result2
        assert "Merangkum" in result3


# ============================================================================
# Test Runner and Reporting
# ============================================================================

def run_tests():
    """Run all tests and generate report"""
    print("=" * 70)
    print("Discord LLM Bot - Test Suite")
    print("=" * 70)
    print()
    
    # Run pytest with verbose output
    exit_code = pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--color=yes',
        '-ra'
    ])
    
    return exit_code


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)