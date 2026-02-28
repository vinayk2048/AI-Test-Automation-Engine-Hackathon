import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class GrokClient:
    def __init__(self):
        self.api_key = os.getenv("GROK_API_KEY")
        self.model = os.getenv("MODEL_NAME", "grok-2-latest")
        self.temperature = float(os.getenv("TEMPERATURE", 0.7))
        self.max_tokens = int(os.getenv("MAX_TOKENS", 2000))

        if not self.api_key:
            raise ValueError("GROK_API_KEY not found in environment variables.")

        self.client = Groq(api_key=self.api_key)

    def generate_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response.choices[0].message.content