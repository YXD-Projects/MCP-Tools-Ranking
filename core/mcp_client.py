from contextlib import AsyncExitStack
from typing import Optional
from mcp import ClientSession
from services.groq_service import GroqService
from services.tool_service import ToolService
from core.server_connection import connect_to_server
from core.query_processor import process_query

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.groq = GroqService()
        self.tool_service = ToolService()

    async def connect_to_server(self):
        self.session = await connect_to_server(self.exit_stack)

    async def process_query(self, query: str) -> str:
        return await process_query(query, self.session, self.groq)

    async def chat_loop(self):
        await chat_loop(self)

    async def cleanup(self):
        await self.exit_stack.aclose()

async def chat_loop(client):
    print("\nMCP Client Started!")
    print("Type your queries or 'quit' to exit.")
    while True:
        try:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break
            response = await client.process_query(query)
            print("\n" + response)
        except Exception as e:
            print(f"\nError: {str(e)}")
