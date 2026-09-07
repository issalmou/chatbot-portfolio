"""Interface commune pour les providers d'embedding.

Totalement indépendante de app/llm/ (fallback de génération Gemini/Mistral/
Groq/OpenAI) : changer de modèle d'embedding ne doit jamais toucher au code
de génération, et inversement. Voir app/embeddings/e5_provider.py pour
l'implémentation active (E5 multilingue local).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    model_id: str
    model_version: str

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimension réelle des vecteurs produits (vérifiée sur le modèle
        chargé, jamais supposée)."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeddings pour l'indexation (chunks de translations.js)."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embedding pour une question utilisateur."""
