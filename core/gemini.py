from __future__ import annotations

import json
from typing import Any, Dict, Optional

from google import genai
from google.genai import types


class GeminiService:
    """
    Gemini service used by FinanceFlow AI.

    Gemini is responsible for:
    - document understanding
    - structured extraction
    - financial explanations
    - management report generation

    Deterministic financial calculations are NOT delegated
    to Gemini.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.8-flash"
    ):
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = model

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Generate normal text from Gemini.
        """

        config_kwargs = {}

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                **config_kwargs
            )
        )

        if not response or not response.text:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return response.text.strip()

    def generate_json(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate structured JSON from Gemini.

        The JSON schema is supplied to Gemini so that
        downstream agents receive predictable data.
        """

        config_kwargs = {
            "response_mime_type": "application/json",
            "response_schema": schema,
        }

        if system_instruction:
            config_kwargs["system_instruction"] = (
                system_instruction
            )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                **config_kwargs
            )
        )

        if not response or not response.text:
            raise RuntimeError(
                "Gemini returned an empty JSON response."
            )

        raw = response.text.strip()

        # Remove accidental markdown code fences.
        if raw.startswith("```"):
            raw = raw.replace("```json", "")
            raw = raw.replace("```", "")
            raw = raw.strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Gemini returned invalid JSON: {raw[:500]}"
            ) from exc

        if not isinstance(result, dict):
            raise RuntimeError(
                "Gemini JSON response was not an object."
            )

        return result
