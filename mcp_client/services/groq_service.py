from groq import Groq
from mcp_client.config.settings import MODEL_NAME

class GroqService:
    def __init__(self):
        self.client = Groq()

    @property
    def chat(self):
        return self.client.chat

    def create_completion(self, messages, tools=None):
        return self.client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=1000,
            messages=messages,
            tools=tools
        )
