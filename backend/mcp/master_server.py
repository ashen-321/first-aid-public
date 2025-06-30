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
import os
import sys

module_paths = ["./", "../"]
file_path = os.path.dirname(__file__)
os.chdir(file_path)

for module_path in module_paths:
    full_path = os.path.normpath(os.path.join(file_path, module_path))
    sys.path.append(full_path)

from mcp_client import MCPClient
from master_server_client import MasterServerClient
from orchestration import Orchestration
from agents import set_agent_config


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastMCP server
app = FastMCP("master_server")

# Initialize orchestration graph
orch = Orchestration()

# @app.tool(description="Return num1 to the power of num2")
# async def any_exponent(num1: int = 2, num2: int = 10) -> str:
#     logging.info(f"Finding {num1}^{num2}")
    
#     """
#     Return num1 to the power of num2

#     Args:
#         num1: the base to raise by num2
#         num1: the exponent to raise num1 to

#     Returns:
#         String representing answer numerically
#     """

#     tool_args = {'num1': num1, 'num2': num2}
    
#     if num1 % 2:
#         result = await master_server_client.call_tool('test_server_1', 'odd_exponent', tool_args)
#     else:
#         result = await master_server_client.call_tool('test_server_2', 'even_exponent', tool_args)

#     # Process and return response
#     logging.info(f'Result: {result}')
#     return result.content[0].text

# @app.tool(description="Automatically call MCP servers and access tools depending on the input query.")
# @app.tool(description="Use this tool to find the exponent between two numbers. Input the base, then a carrot, then the exponent in a single string.")
@app.tool()
async def access_sub_mcp(query: str):
    logging.info(f'Collecting tool information for query: {query}')

    set_agent_config({
        "tools": master_server_client.available_tools_flattened,
        "master_server_client": master_server_client
    })

    result = await orch.graph.ainvoke(
        {"question": query},
        {"recursion_limit": 30},
    )
    logging.info(f'Orchestration result: {result}')

    return result.get('external_data')

logging.info(app._tool_manager._tools)


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can serve the provided mcp server with SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,
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
    
    master_server_client = MasterServerClient(app)

    try:
        # await master_server_client.connect_to_server("http://localhost:8091/mcp", 'test_server_1')
        # await master_server_client.connect_to_server("http://localhost:8092/mcp", 'test_server_2')
        await master_server_client.connect_to_server("http://localhost:8093/mcp", 'finance_server')

        # await master_server_client.connect_to_server("http://agent.cavatar.info:8080/mcp", 'github')
        # await master_server_client.connect_to_server("http://infs.cavatar.info:8084/mcp", 'arxiv')
        # await master_server_client.connect_to_server("http://infs.cavatar.info:8083/mcp", 'medrxiv')
        
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
