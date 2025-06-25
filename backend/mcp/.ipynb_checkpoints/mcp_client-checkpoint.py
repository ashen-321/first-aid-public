from typing import Optional
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from openai import OpenAI
from openai.types import Completion
import os
import httpx

model_id_c37 = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
model_id_nova = "us.amazon.nova-lite-v1:0"
model_id_llama = "meta.llama3-3-70b-instruct-v1:0"

os.environ["OPENAI_API_KEY"] = api_key = os.getenv("bedrock_api_token")
os.environ["OPENAI_BASE_URL"] = base_url = os.getenv("bedrock_api_url") 

class MCPClient:
    """MCP Client for interacting with an MCP Streamable HTTP server"""

    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.openai = OpenAI()
        self.available_tools = []

    async def connect_to_streamable_http_server(self, server_url: str, headers: Optional[dict] = None):
        """Connect to an MCP server running with HTTP Streamable transport"""
        self._streams_context = streamablehttp_client(  # pylint: disable=W0201
            url=server_url,
            headers=headers or {},
        )
        read_stream, write_stream, _ = await self._streams_context.__aenter__()  # pylint: disable=E1101

        self._session_context = ClientSession(read_stream, write_stream)  # pylint: disable=W0201
        self.session: ClientSession = await self._session_context.__aenter__()  # pylint: disable=C2801

        await self.session.initialize()

    async def get_available_tools(self):
        """Get available tools from the server"""
        print("Fetching available server tools...")
        response = await self.session.list_tools()
        print("Connected to MCP server with tools:", [tool.name for tool in response.tools])

        # Format tools for OpenAI
        available_tools = [
            {
                "type": 'function',
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
                "strict": True,
            }
            for tool in response.tools
        ]
        self.available_tools = available_tools

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [{"role": "user", "content": query}]

        await self.get_available_tools()

        # Initial OpenAI API call
        response = self.openai.chat.completions.create(
            model=model_id_nova,
            max_tokens=3000,
            messages=messages,
            tools=self.available_tools
        )

        # Handle tool calls
        final_text = []
        tool_calls = response.choices[0].message.tool_calls

        if tool_calls:
            for call in tool_calls:
                tool_name = call.function.name
                tool_args = eval(call.function.arguments)
                
                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                print(f"[Calling tool {tool_name} with args {tool_args} to get {result}]")
    
                # Continue conversation with tool results
                if result.isError:
                    print(f"Tool call failed with error {result.content[0].text}")
                else:
                    messages.append({"role": "user", "content": result.content[0].text})

        # Get next response from LLM
        response = self.openai.chat.completions.create(
            model=model_id_nova,
            max_tokens=3000,
            messages=messages,
        )

        final_text.append(response.choices[0].message.content)

        # Compile and return output
        print('messages', messages)
        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        # Exit if connection failed
        if not self.session:
            return

        # Run query chat loop
        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Properly clean up the session and streams"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:  # pylint: disable=W0125
            await self._streams_context.__aexit__(None, None, None)  # pylint: disable=E1101