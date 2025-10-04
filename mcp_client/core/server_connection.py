from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp_client.config.settings import DOCKER_COMMAND, SERVER_ARGS, SERVER_ENV

async def connect_to_server(exit_stack):
    server_params = StdioServerParameters(
        command=DOCKER_COMMAND,
        args=SERVER_ARGS,
        env=SERVER_ENV
    )
    stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
    stdio, write = stdio_transport
    session = await exit_stack.enter_async_context(ClientSession(stdio, write))
    await session.initialize()
    response = await session.list_tools()
    print(response)
    print("\nConnected to server with tools:", [tool.name for tool in response.tools])

    return session
