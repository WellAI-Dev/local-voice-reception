"""LLM interface module."""

from .knowledge import KnowledgeManager
from .ollama_client import OllamaClient

__all__ = ["KnowledgeManager", "OllamaClient"]
