import json
import os

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

class GeminiService:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        try:
            from google import genai
        except ImportError as e:
            raise RuntimeError("google-genai is missing. Install requirements.txt before running FinanceFlow AI.") from e
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is not configured.")
        self.client = genai.Client(api_key=key)
        self.model = model or DEFAULT_MODEL

    def generate_json(self, prompt: str, schema: dict):
        from google.genai import types
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        text = response.text or "{}"
        return json.loads(text)

    def generate_text(self, prompt: str):
        from google.genai import types
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3),
        )
        return response.text or ""
