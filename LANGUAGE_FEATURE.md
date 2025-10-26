# Indonesian Language Support

This document describes the Indonesian language feature added to the Discord LLM Summarization Bot.

## Overview

The bot now supports generating summaries in both **English** (default) and **Indonesian** (Bahasa Indonesia). Users can choose their preferred language when requesting summaries.

## How to Use

### `/catchup` Command

Add the optional `language` parameter:

```bash
/catchup                          # English (default)
/catchup language:Indonesian      # Indonesian
```

### `/summarize` Command

Add the optional `language` parameter to any summarization request:

```bash
# Time-based summaries
/summarize mode:2h language:Indonesian
/summarize mode:1d detail:detailed language:Indonesian

# Context-based summaries
/summarize mode:12h context:meeting language:Indonesian
/summarize mode:1d context:project deadline language:Indonesian

# Since last seen
/summarize mode:catchup language:Indonesian
```

## What Gets Translated

When you select Indonesian language:

1. **Summary Content**: The actual summary of messages is generated in Indonesian
2. **System Messages**: Error messages and status messages appear in Indonesian
3. **Metadata**: Information like "Merangkum X pesan dari Y peserta" (Summarized X messages from Y participants)

## Examples

### English Summary (Default)
```
User: /summarize mode:2h

Bot: The main topics discussed were:
- Project deadline extension to next week
- New feature requirements for the dashboard
- Team meeting scheduled for Friday at 2 PM

Key decisions:
- Approved the timeline extension
- Sarah will lead the dashboard feature

*Summarized 45 messages from 8 participants*
```

### Indonesian Summary
```
User: /summarize mode:2h language:Indonesian

Bot: Topik utama yang dibahas:
- Perpanjangan deadline proyek hingga minggu depan
- Kebutuhan fitur baru untuk dashboard
- Rapat tim dijadwalkan Jumat jam 2 siang

Keputusan penting:
- Menyetujui perpanjangan timeline
- Sarah akan memimpin fitur dashboard

*Merangkum 45 pesan dari 8 peserta*
```

## Implementation Details

### Files Modified

1. **utils/llm_handler.py**
   - Added `language` parameter to `create_summary_prompt()`
   - Indonesian prompt templates for all summary types (general, detailed, brief, context)
   - Localized error messages and metadata

2. **utils/summarizer.py**
   - Added `language` parameter to all methods:
     - `summarize_since_last_seen()`
     - `summarize_time_range()`
     - `summarize_with_context()`
     - `quick_catchup()`
   - Localized status messages
   - Indonesian time formatting in `_format_time_ago()`

3. **bot.py**
   - Added language choice parameter to `/summarize` command
   - Added language choice parameter to `/catchup` command
   - Updated help text to include language examples

4. **Documentation**
   - Updated README.md with language feature
   - Updated QUICKSTART.md with language examples
   - Updated in-bot help command (`/summary_help`)

## Language Detection

The bot accepts these values for Indonesian:
- `indonesian` (case-insensitive)
- `indonesia` (case-insensitive)

Any other value defaults to English.

## Prompt Engineering

Indonesian prompts are carefully crafted to:
- Use natural Bahasa Indonesia
- Match the tone and style of English prompts
- Preserve the same level of detail
- Maintain consistency across different summary types

## Future Enhancements

Potential improvements:
- Auto-detect language from channel/server settings
- Add more languages (Spanish, Japanese, etc.)
- Per-user language preferences stored in database
- Language-specific prompt customization
- Mixed-language message handling

## Testing

To test the Indonesian feature:

1. Send messages in a Discord channel
2. Use `/summarize mode:2h language:Indonesian`
3. Verify the summary is in Indonesian
4. Check that metadata and error messages are also in Indonesian

## Cost Implications

There is **no additional cost** for Indonesian summaries. The token usage is approximately the same as English summaries, as the OpenAI models handle multiple languages efficiently.

## Support

If you encounter issues with Indonesian summaries:
- Check that the language parameter is set correctly
- Verify your OpenAI API key has sufficient credits
- Review console logs for any errors
- The bot will still function normally if language parameter is omitted (defaults to English)
