from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import MessagesState, END
from langgraph.types import Command
from typing import Literal



async def mcp_nodes(state: MessagesState) -> Command[Literal[END]]:
    print(f"In mcp_nodes: {state}\n=======\n")
    async with MultiServerMCPClient(
        {
            "math": {
                "command": "/home/aaron/.conda/envs/firstaid/bin/python",
                "args": ["/home/aaron/src/first-aid/scripts/mcp_math_server.py"],
                "transport": "stdio",
            },
            "weather": {
                # make sure you start your weather server on port 8000
                # python ~/src/first-aid/scripts/mcp_weather_server.py &
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            },
            "timezone": {
                "command": "/home/aaron/.conda/envs/firstaid/bin/python",
                "args": ["/home/aaron/src/first-aid/scripts/mcp_tz_inlineagent.py"],
                "transport": "stdio",
            }
        }
    ) as client:
        agent = create_react_agent(llm, client.get_tools())
        # Add system message
        system_message = SystemMessage(content=(
                "You have access to multiple tools that can help answer queries. "
                "Use them dynamically and efficiently based on the user's request. "
        ))
        
        result = await agent.ainvoke(state)
        return Command(
            update={
                # share internal message history of chart agent with other agents
                "messages": result["messages"][-1].content,
            },
        )