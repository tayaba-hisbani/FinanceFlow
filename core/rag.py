from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PolicyRAG:
    """
    Lightweight local RAG system for FinanceFlow AI.

    Loads company policy, creates TF-IDF embeddings,
    and retrieves the most relevant policy sections.
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

            project_root = Path(__file__).resolve().parent.parent

            alternative_path = (
                project_root
                / "data"
                / "company_policy.txt"
            )

            if alternative_path.exists():
                self.policy_path = alternative_path
            else:
                raise FileNotFoundError(
                    f"Company policy file not found: "
                    f"{self.policy_path}"
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

        self.matrix = self.vectorizer.fit_transform(
            self.chunks
        )

    @staticmethod
    def _split_text(text: str) -> List[str]:
        """
        Split the policy into sections.
        Markdown headings create new chunks.
        """

        sections = []
        current = []

        for line in text.splitlines():

            line = line.strip()

            if not line:
                continue

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
        k: Optional[int] = None,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve the most relevant policy sections.

        Supports BOTH:
            retrieve(query, k=2)
        and:
            retrieve(query, top_k=2)

        This prevents parameter-name conflicts between
        different agents.
        """

        if not query or not query.strip():
            return []

        # Support both parameter names.
        if k is not None:
            number_to_return = k
        elif top_k is not None:
            number_to_return = top_k
        else:
            number_to_return = 3

        number_to_return = max(1, int(number_to_return))

        if self.matrix is None:
            self._load_policy()

        query_vector = self.vectorizer.transform(
            [query]
        )

        similarities = cosine_similarity(
            query_vector,
            self.matrix
        )[0]

        ranked_indices = similarities.argsort()[::-1]

        results = []

        for index in ranked_indices[:number_to_return]:

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
        k: Optional[int] = None,
        top_k: Optional[int] = None
    ) -> List[str]:
        """Return only the retrieved policy text."""

        results = self.retrieve(
            query=query,
            k=k,
            top_k=top_k
        )

        return [
            result["text"]
            for result in results
        ]

    def get_context(
        self,
        query: str,
        k: Optional[int] = None,
        top_k: Optional[int] = None
    ) -> str:
        """
        Return retrieved policy evidence as a single
        context string for Gemini.
        """

        results = self.retrieve(
            query=query,
            k=k,
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
