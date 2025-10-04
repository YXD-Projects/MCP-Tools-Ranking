import json

import httpx

from config.settings import SERVER_QUERY_URL


async def process_query(query: str, session, groq):
    messages = [{"role": "user", "content": query}]
    async with httpx.AsyncClient() as client:
        gateway_response = await client.post(
            SERVER_QUERY_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps({"query": query})
        )
        gateway_data = gateway_response.json()
        print("Gateway response:", gateway_data)

    response = await session.list_tools()

    available_tools = [{
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema
        }
    } for tool in response.tools]

    print(available_tools)
    print("length of tools:", len(available_tools))

    response = "Mock answer from the LLM"
    final_text = [response]
    """for content in response.content:
        if content.type == 'text':
            final_text.append(content.text)
        elif content.type == 'tool_use':
            tool_name = content.name
            tool_args = content.input
            result = await session.call_tool(tool_name, tool_args)
            final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

            if hasattr(content, 'text') and content.text:
                messages.append({"role": "assistant", "content": content.text})
            messages.append({"role": "user", "content": result.content})

            response = groq.create_completion(messages)
            final_text.append(response.content[0].text)"""

    return "\n".join(final_text)
