import json

import httpx
from mcp_client.config.settings import SERVER_QUERY_URL


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

    response = groq.create_completion(messages, available_tools)
    final_text = []

    assistant_message = response.choices[0].message
    if assistant_message.content:
        final_text.append(assistant_message.content)
    if assistant_message.tool_calls:
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            result = await session.call_tool(tool_name, tool_args)
            final_text.append(f"[Called tool {tool_name} with args {tool_args}]")

            # Add tool result back into conversation
            messages.append({"role": "assistant", "content": assistant_message.content})
            messages.append({"role": "user", "content": result.content})

            # Re-ask Groq after tool output
            follow_up = groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
            )
            final_text.append(follow_up.choices[0].message.content)

    return "\n".join(final_text)
