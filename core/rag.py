from __future__ import annotations

from pathlib import Path
from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PolicyRAG:
    """
    Lightweight local RAG system for FinanceFlow AI.

    It:
    1. Loads the company policy document.
    2. Splits it into useful chunks.
    3. Creates TF-IDF vectors.
    4. Retrieves the most relevant policy sections for a query.

    This keeps the hackathon MVP simple and avoids requiring
    a separate vector database.
    """

    def __init__(self, policy_path: str = "data/company_policy.txt"):
        self.policy_path = Path(policy_path)

        self.chunks: List[str] = []
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
        self.matrix = None

        self._load_policy()

    def _load_policy(self) -> None:
        """Load and index the company policy."""

        if not self.policy_path.exists():
            # Try an alternative path relative to this file.
            project_root = Path(__file__).resolve().parent.parent
            alternative_path = project_root / "data" / "company_policy.txt"

            if alternative_path.exists():
                self.policy_path = alternative_path
            else:
                raise FileNotFoundError(
                    f"Company policy file not found: {self.policy_path}"
                )

        text = self.policy_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        self.chunks = self._split_text(text)

        if not self.chunks:
            raise ValueError(
                "Company policy file is empty."
            )

        self.matrix = self.vectorizer.fit_transform(self.chunks)

    @staticmethod
    def _split_text(text: str) -> List[str]:
        """
        Split policy into meaningful sections.

        The policy file is structured with headings such as:
        ## Invoice Approval
        ## Duplicate Invoices
        etc.
        """

        sections = []
        current = []

        for line in text.splitlines():

            line = line.strip()

            if not line:
                continue

            # Treat markdown headings as section boundaries.
            if line.startswith("#") and current:
                sections.append(" ".join(current))
                current = []

            current.append(line)

        if current:
            sections.append(" ".join(current))

        return sections

    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Retrieve the most relevant policy sections.

        Returns:
            [
                {
                    "text": "...",
                    "score": 0.82
                }
            ]
        """

        if not query or not query.strip():
            return []

        if self.matrix is None:
            self._load_policy()

        query_vector = self.vectorizer.transform([query])

        similarities = cosine_similarity(
            query_vector,
            self.matrix
        )[0]

        ranked_indices = similarities.argsort()[::-1]

        results = []

        for index in ranked_indices[:top_k]:

            score = float(similarities[index])

            results.append(
                {
                    "text": self.chunks[index],
                    "score": round(score, 4)
                }
            )

        return results

    def search(
        self,
        query: str,
        top_k: int = 3
    ) -> List[str]:
        """
        Convenience method that returns only policy text.
        """

        results = self.retrieve(
            query=query,
            top_k=top_k
        )

        return [
            result["text"]
            for result in results
        ]

    def get_context(
        self,
        query: str,
        top_k: int = 3
    ) -> str:
        """
        Return retrieved policy sections as one context string
        for Gemini.
        """

        results = self.retrieve(
            query=query,
            top_k=top_k
        )

        if not results:
            return "No relevant company policy was found."

        context_parts = []

        for i, result in enumerate(results, start=1):

            context_parts.append(
                f"[Policy Evidence {i} | "
                f"similarity={result['score']}]\n"
                f"{result['text']}"
            )

        return "\n\n".join(context_parts)
