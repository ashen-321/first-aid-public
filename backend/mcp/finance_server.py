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

module_paths = ["./", "../scripts"]
file_path = os.path.dirname(__file__)
os.chdir(file_path)

for module_path in module_paths:
    full_path = os.path.normpath(os.path.join(file_path, module_path))
    sys.path.append(full_path)

from reasoning_finance_advisor import reasoning_finance_team


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastMCP server
app = FastMCP("finance")
started: bool = False

@app.tool(description="Get finance and stock information using up-to-date information via a team of AI agents.")
async def finance_request(query: str) -> str:
    logging.info(f"Getting finance recommendation for query '{query}'")

    response = reasoning_finance_team.run(query, stream=False).content
    logging.info(f'Response: {response}')
    
    return response

# @app.tool(description="Get finance, stock, and general news-related recommendations using up-to-date information via a team of AI agents.")
# async def finance_request(query: str) -> str:
#     logging.info(f"Getting finance recommendation for query '{query}'")

#     response = next(reasoning_finance_team.run(query, stream=True)).content
#     logging.info(f'Streamed response: {response}')
    
#     yield response

def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can serve the provided mcp server with SSE."""
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

def startup():
    global started
    
    if started:
        return

    started = True
    app.run(transport="streamable-http",host="0.0.0.0",port=8093)

if __name__ == '__main__':
    startup()
