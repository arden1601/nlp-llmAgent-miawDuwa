import openai
from typing import List, Dict, Optional
import os

class LLMHandler:
    """Handles OpenAI API interactions for summarization"""

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"  # Cost-effective model, can change to gpt-4o for better quality

    def create_summary_prompt(
        self,
        messages: List[Dict],
        context: Optional[str] = None,
        summary_type: str = "general",
        language: str = "english"
    ) -> str:
        """Create a prompt for the LLM based on messages and context"""

        # Format messages for the prompt
        formatted_messages = []
        for msg in messages:
            timestamp = msg['timestamp'].split('T')[1][:5]  # Extract HH:MM
            content = msg['content'] if msg['content'] else "[attachment]" if msg['has_attachments'] else "[empty]"
            formatted_messages.append(f"[{timestamp}] {msg['username']}: {content}")

        messages_text = "\n".join(formatted_messages)

        # Language instruction
        lang_instruction = ""
        if language.lower() == "indonesian" or language.lower() == "indonesia":
            lang_instruction = "IMPORTANT: Provide the summary in Indonesian language (Bahasa Indonesia)."

        # Base prompt templates
        if summary_type == "general":
            if language.lower() in ["indonesian", "indonesia"]:
                system_prompt = """Anda adalah asisten yang membantu merangkum percakapan chat Discord.
Berikan ringkasan yang jelas dan terorganisir untuk membantu seseorang mengetahui apa yang terlewatkan.
Sertakan:
- Topik utama yang dibahas
- Keputusan atau kesimpulan penting
- Pengumuman atau informasi penting
- Percakapan atau perdebatan yang menonjol

Buat ringkasan yang ringkas namun informatif."""

                user_prompt = f"""Silakan rangkum pesan-pesan chat Discord berikut:

{messages_text}

Berikan ringkasan yang terstruktur dengan baik."""
            else:
                system_prompt = """You are a helpful assistant that summarizes Discord chat conversations.
Provide a clear, organized summary that helps someone catch up on what they missed.
Include:
- Main topics discussed
- Key decisions or conclusions
- Important announcements or information
- Notable conversations or debates

Keep the summary concise but informative."""

                user_prompt = f"""Please summarize the following Discord chat messages:

{messages_text}

Provide a well-structured summary."""

        elif summary_type == "detailed":
            if language.lower() in ["indonesian", "indonesia"]:
                system_prompt = """Anda adalah asisten yang membuat ringkasan detail dari percakapan Discord.
Berikan ringkasan komprehensif yang diorganisir berdasarkan topik.
Sertakan nama peserta, poin-poin kunci, dan konteks penting."""

                user_prompt = f"""Silakan berikan ringkasan detail dari chat Discord berikut:

{messages_text}

Organisir berdasarkan topik dan sertakan peserta kunci."""
            else:
                system_prompt = """You are a helpful assistant that creates detailed summaries of Discord conversations.
Provide a comprehensive summary organized by topics.
Include participant names, key points, and any important context."""

                user_prompt = f"""Please provide a detailed summary of the following Discord chat:

{messages_text}

Organize by topics and include key participants."""

        elif summary_type == "brief":
            if language.lower() in ["indonesian", "indonesia"]:
                system_prompt = """Anda adalah asisten yang membuat ringkasan singkat dari percakapan Discord.
Berikan hanya highlight yang paling penting dalam beberapa kalimat."""

                user_prompt = f"""Berikan ringkasan singkat (2-3 kalimat) dari pesan Discord ini:

{messages_text}"""
            else:
                system_prompt = """You are a helpful assistant that creates brief summaries of Discord conversations.
Provide only the most important highlights in a few sentences."""

                user_prompt = f"""Provide a brief summary (2-3 sentences) of these Discord messages:

{messages_text}"""

        else:  # context-based
            if language.lower() in ["indonesian", "indonesia"]:
                system_prompt = """Anda adalah asisten yang merangkum percakapan Discord dengan fokus spesifik.
Ekstrak dan rangkum informasi yang relevan dengan konteks atau topik yang diberikan."""

                user_prompt = f"""Silakan rangkum chat Discord berikut, dengan fokus pada: {context}

{messages_text}

Fokus hanya pada pesan yang terkait dengan: {context}"""
            else:
                system_prompt = """You are a helpful assistant that summarizes Discord conversations with a specific focus.
Extract and summarize information relevant to the given context or topic."""

                user_prompt = f"""Please summarize the following Discord chat, focusing on: {context}

{messages_text}

Focus only on messages related to: {context}"""

        return system_prompt, user_prompt

    async def generate_summary(
        self,
        messages: List[Dict],
        context: Optional[str] = None,
        summary_type: str = "general",
        max_tokens: int = 500,
        language: str = "english"
    ) -> str:
        """Generate a summary using OpenAI API"""

        if not messages:
            if language.lower() in ["indonesian", "indonesia"]:
                return "Tidak ada pesan untuk dirangkum."
            return "No messages to summarize."

        system_prompt, user_prompt = self.create_summary_prompt(messages, context, summary_type, language)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )

            summary = response.choices[0].message.content.strip()

            # Add metadata
            message_count = len(messages)
            participants = len(set(msg['username'] for msg in messages))

            if language.lower() in ["indonesian", "indonesia"]:
                metadata = f"\n\n*Merangkum {message_count} pesan dari {participants} peserta*"
            else:
                metadata = f"\n\n*Summarized {message_count} messages from {participants} participants*"

            return summary + metadata

        except Exception as e:
            if language.lower() in ["indonesian", "indonesia"]:
                return f"Error saat membuat ringkasan: {str(e)}"
            return f"Error generating summary: {str(e)}"

    async def generate_quick_catchup(self, messages: List[Dict], language: str = "english") -> str:
        """Generate a quick bullet-point catchup"""

        if not messages:
            if language.lower() in ["indonesian", "indonesia"]:
                return "Tidak ada pesan baru sejak Anda terakhir aktif."
            return "No new messages since you were last active."

        formatted_messages = []
        for msg in messages:
            content = msg['content'] if msg['content'] else "[attachment]"
            formatted_messages.append(f"{msg['username']}: {content}")

        messages_text = "\n".join(formatted_messages)

        try:
            if language.lower() in ["indonesian", "indonesia"]:
                system_content = "Buat ringkasan poin-poin penting dari pesan Discord. Gunakan maksimal 3-5 poin. Ringkas dan jelas."
                user_content = f"Rangkum pesan-pesan berikut:\n\n{messages_text}"
            else:
                system_content = "Create a quick bullet-point summary of Discord messages. Use 3-5 bullet points maximum. Be concise."
                user_content = f"Summarize these messages:\n\n{messages_text}"

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_content
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ],
                max_tokens=200,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            if language.lower() in ["indonesian", "indonesia"]:
                return f"Error saat membuat catchup: {str(e)}"
            return f"Error generating catchup: {str(e)}"
