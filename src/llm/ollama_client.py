"""
Ollama LLM Client.
Provides interface to local Ollama LLM for response generation.
"""

import logging
from typing import TYPE_CHECKING, Generator, List, Optional

import ollama
from ollama import Client

if TYPE_CHECKING:
    from src.llm.knowledge import KnowledgeManager

logger = logging.getLogger(__name__)

# Default system prompt for reception AI
DEFAULT_SYSTEM_PROMPT = """あなたは株式会社WellAI（ウェルアイ）の電話受付AIアシスタントです。
お客様からのお問い合わせに対して、丁寧かつ明るい親しみやすいトーンで回答してください。

回答のルール:
- 丁寧語・謙譲語を使用してください
- 明るく親しみやすいトーンで話してください
- 回答は3文以内に簡潔にまとめてください
- 不明な点は「確認して担当者より折り返しご連絡いたします」と伝えてください
- 個人情報は取り扱わないでください"""


class OllamaClient:
    """
    Ollama LLM client for local language model inference.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        knowledge_manager: Optional["KnowledgeManager"] = None,
    ):
        """
        Initialize Ollama client.

        Args:
            model: Model name (e.g., "qwen2.5:7b", "gemma2:9b")
            base_url: Ollama server URL
            system_prompt: System prompt for the AI
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            knowledge_manager: Optional KnowledgeManager for auto-injecting
                knowledge context into prompts
        """
        self.model = model
        self.base_url = base_url
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.knowledge_manager = knowledge_manager

        self.client = Client(host=base_url)
        self._conversation_history: List[dict] = []

        logger.info(f"Ollama client initialized: model={model}, url={base_url}")

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        use_history: bool = True,
    ) -> str:
        """
        Generate a response for the given prompt.

        Args:
            prompt: User's input text
            context: Optional RAG context to include
            use_history: Whether to include conversation history

        Returns:
            Generated response text
        """
        messages = self._build_messages(prompt, context, use_history)

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )

            assistant_message = response["message"]["content"]

            # Update history
            if use_history:
                self._conversation_history.append({"role": "user", "content": prompt})
                self._conversation_history.append(
                    {"role": "assistant", "content": assistant_message}
                )

            return assistant_message

        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise

    def generate_stream(
        self,
        prompt: str,
        context: Optional[str] = None,
        use_history: bool = True,
    ) -> Generator[str, None, None]:
        """
        Generate a streaming response.

        Args:
            prompt: User's input text
            context: Optional RAG context
            use_history: Whether to include conversation history

        Yields:
            Response tokens as they are generated
        """
        messages = self._build_messages(prompt, context, use_history)
        full_response = ""

        try:
            stream = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
                stream=True,
            )

            for chunk in stream:
                token = chunk["message"]["content"]
                full_response += token
                yield token

            # Update history after complete response
            if use_history:
                self._conversation_history.append({"role": "user", "content": prompt})
                self._conversation_history.append(
                    {"role": "assistant", "content": full_response}
                )

        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise

    def _build_messages(
        self,
        prompt: str,
        context: Optional[str],
        use_history: bool,
    ) -> List[dict]:
        """Build message list for Ollama API."""
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history
        if use_history:
            messages.extend(self._conversation_history)

        # Auto-inject knowledge context if no explicit context provided
        effective_context = context
        if effective_context is None and self.knowledge_manager is not None:
            effective_context = self.knowledge_manager.get_context()

        # Build user message with optional context
        if effective_context:
            user_content = f"""以下の情報を参考にして回答してください。

【参考情報】
{effective_context}

【質問】
{prompt}"""
        else:
            user_content = prompt

        messages.append({"role": "user", "content": user_content})

        return messages

    def clear_history(self):
        """Clear conversation history."""
        self._conversation_history = []
        logger.info("Conversation history cleared")

    def check_connection(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            models = self.client.list()
            return True
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            return False

    def is_model_available(self) -> bool:
        """Check if the configured model is available."""
        try:
            models = self.client.list()
            model_names = [m["name"] for m in models.get("models", [])]
            # Check for exact match or partial match (e.g., "qwen2.5:7b" in "qwen2.5:7b")
            return any(
                self.model in name or name.startswith(self.model.split(":")[0])
                for name in model_names
            )
        except Exception:
            return False


if __name__ == "__main__":
    # Test LLM
    logging.basicConfig(level=logging.INFO)

    client = OllamaClient()

    if not client.check_connection():
        print("Error: Ollama server not running. Start with: ollama serve")
        exit(1)

    if not client.is_model_available():
        print(f"Error: Model {client.model} not available. Pull with: ollama pull {client.model}")
        exit(1)

    print("Testing Ollama client...")
    response = client.generate("こんにちは、営業時間を教えてください。")
    print(f"Response: {response}")
