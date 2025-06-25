from typing import Any, List, Dict, Optional
import asyncio
import logging
from fastmcp import FastMCP
from mcp.server import Server
from starlette.requests import Request
from mcp.server.fastmcp.prompts import base
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.routing import Mount, Route
import uvicorn

from mcp_client import MCPClient
from master_server_client import MasterServerClient


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastMCP server
app = FastMCP("master_server")


def route_query(query: str):
    print(f'\n--- Selecting tools ---\n')
    
    tool_selection = ["github", "arxiv", "medrxiv"]

    llm = ChatOpenAI(
        model=config['model_id'],
        temperature=1.0,
        max_tokens=None,
        timeout=None,
        max_retries=5,
    )

    messages = [
        ("system", 
            'You are a seasoned medical expert. Your role is to determine the specificity of the query and decide whether or not online information is sufficient to resolve the query. Respond with either a "YES", "NO", or "UNCLEAR" based on if online information is adequate. Do not output anything else besides those three options. Output in all capital letters.'
        ),
        ("human", state['rewritten_question']),
    ]

    response = llm.invoke(messages).content
    print(f'Tools router response: {response}')
    
    if "YES" in response:
        tool_selection = ["web_search_node"]
    elif "NO" in response:
        tool_selection = ["pubmed_node", "medrxiv_node"]
    else:
        tool_selection = ["pubmed_node"]
    # Use all three tools if unclear

    return {'tools_requested': tool_selection}

@app.tool(description="Search github, arxiv, and medrxiv")
async def mcp_search(query: str) -> str:
    logging.info(f"Querying MCP servers...")

    

# @app.tool(description="Return num1 to the power of num2")
async def any_exponent(num1: int = 2, num2: int = 10) -> str:
    logging.info(f"Finding {num1}^{num2}")
    
    """
    Return num1 to the power of num2

    Args:
        num1: the base to raise by num2
        num1: the exponent to raise num1 to

    Returns:
        String representing answer numerically
    """

    tool_args = {'num1': num1, 'num2': num2}
    
    if num1 % 2:
        logging.info(f"[Calling tool odd_exponent from server 1 with args {tool_args}]")
        result = await master_server_client.call_tool('test_server_1', 'odd_exponent', tool_args)
    else:
        logging.info(f"[Calling tool even_exponent from server 2 with args {tool_args}]")
        result = await master_server_client.call_tool('test_server_2', 'even_exponent', tool_args)

    # Process and return response
    logging.info(f'Result: {result}')
    return result.content[0].text


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can server the provied mcp server with SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


# Server-server communications
master_server_client = None

async def initialize_interserver_comms():
    global master_server_client
    
    master_server_client = MasterServerClient()

    try:
        # await master_server_client.connect_to_server("http://localhost:8091/mcp", 'test_server_1')
        # await master_server_client.connect_to_server("http://localhost:8092/mcp", 'test_server_2')

        await master_server_client.connect_to_server("http://agent.cavatar.info:8080/mcp", 'github')
        await master_server_client.connect_to_server("http://infs.cavatar.info:8084/mcp", 'arxiv')
        await master_server_client.connect_to_server("http://infs.cavatar.info:8083/mcp", 'medrxiv')
        
        await master_server_client.server_loop()
    finally:
        await master_server_client.cleanup()
        pass

async def start_app():
    await app.run_async(transport="streamable-http",host="0.0.0.0",port=8089)

async def main():
    # await initialize_interserver_comms()
    await asyncio.gather(initialize_interserver_comms(), start_app())

asyncio.run(main())
