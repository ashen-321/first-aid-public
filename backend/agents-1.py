from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import MessagesState, END
from langgraph.types import Command
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from typing import TypedDict, Literal
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from bokeh.models import ColumnDataSource
from bokeh.plotting import gmap
from bokeh.models import GMapOptions
from bokeh.io import show, output_notebook, export_png
import pandas as pd
import os
import json
import re

import tools
from agent_protocol import MultiAgentState


# --------------------------------------------------------------------------------------------
# Config -------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------


config = {
    'model_id': 'us.anthropic.claude-3-5-haiku-20241022-v1:0',
    'internal_file_path': ''
}

def set_agent_config(options: dict):
    for key in options:
        config[key] = options[key]


# --------------------------------------------------------------------------------------------
# Testing Nodes ------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------


def basic_chat(query: str):
    llm = ChatOpenAI(
        model=config['model_id'],
        temperature=1.0,
        max_tokens=None,
        timeout=None,
        max_retries=5,
    )

    return llm.invoke(query).content

def math_tool_test(query: str):
    llm = ChatOpenAI(
        model=config['model_id'],
        temperature=1.0,
        max_tokens=None,
        timeout=None,
        max_retries=5,
    )
    llm_with_tools = llm.bind_tools([tools.add, tools.multiply])

    # messages = [HumanMessage(query)]
    # response = llm_with_tools.invoke(messages)
    # messages.append(response)

    # for tool_call in response.tool_calls:
    #     selected_tool = {"add": tools.add, "multiply": tools.multiply}[tool_call["name"].lower()]
    #     tool_output = selected_tool.invoke(tool_call["args"])
    #     messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

    # return llm_with_tools.invoke(messages)

    response = llm_with_tools.invoke(query)
    return response.tool_calls

# Define variables
question_category_prompt = """You are a planner agent who decides which team to dispatch to based on the user's input. 
Depending on your answer, question will be routed to the right team, so your task is crucial. 
There are 3 possible question types:
- mcp_client - Answer questions related to simple math operataions, weather inquery or timezone convertion.
- sub_graph_maps - Answer questions about maps related topic.
- perplexity_client - Answer questions required web search for timely and common information.
Return in the output only one word (mcp_client, sub_graph_maps, sub_graph_research or perplexity_client)
"""
rewrite_prompt = """You are a Prompt Optimization Assistant specialized in refining and enhancing user-provided prompts. Your expertise lies in:
- Clarifying ambiguous instructions
- Maintaining the original intent while enhancing clarity
When a user shares a prompt with you, analyze it carefully, then provide an optimized version that preserves their core objective while making it more precise, comprehensive, and effective.
Output only your answer in text string format.
"""
GMAPS_API_KEY = os.getenv("gmaps_api_token")
PERPLEXITY_API_KEY = os.getenv('openperplex_api_token')
google_maps_server_params = StdioServerParameters(
    command="docker",
    args=[
        "run",
        "-i",
        "--rm",
        "-e",
        "GOOGLE_MAPS_API_KEY",
        "mcp/google-maps"
      ],
    env={"GOOGLE_MAPS_API_KEY": GMAPS_API_KEY},
)
perplexity_server_params = StdioServerParameters(
    command="docker",
    args=["run", "-i", "--rm", "-e", "PERPLEXITY_API_KEY", "mcp/perplexity-ask"],
    env={"PERPLEXITY_API_KEY": PERPLEXITY_API_KEY},
)
llm = ChatOpenAI(
    model=config['model_id'],
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=5,
    api_key=os.getenv("bedrock_api_token"),
    base_url=os.getenv("bedrock_api_url")
)

# define router
def rewrite_node(state: MessagesState):
    print(f"Here: {state['messages'][0].content}\n\n")
    messages = [
        SystemMessage(content=rewrite_prompt), 
        HumanMessage(content=state['messages'][0].content)
    ]

    result = llm.invoke(messages)
    state['messages'][0].content = result.content
    #goto = get_next_node(result, router_dispatcher)
    print(f"NOW state after rewite: {result.content}\n\n")
    return Command(
        update={
            # share internal message history of chart agent with other agents
            "messages": result.content,
        },
        #goto=goto,
    )

def router_dispatcher(state: MessagesState):
    print(f"Before router_dispatcher: {state}\n\n")
    print(f"..and message: {state['messages']}\n\n")
    messages = [
        SystemMessage(content=question_category_prompt), 
        HumanMessage(content=state['messages'][-1].content)
    ]

    result = llm.invoke(messages)
    print(f"After router_dispatcher: {result}\n\n")
    print(f"... and type: {type(result)}\n\n")
    if "mcp_client" in result.content:
        return "mcp_client"
    elif "sub_graph_maps" in result.content:
        print("Dispatching to google maps MCP...")
        return "sub_graph_maps"
    elif "perplexity_client" in result.content:
        return "perplexity_client"
    else:
        return END

async def gmaps_node(state: MessagesState) -> Command[Literal["map_plotter", END]]:
    async with stdio_client(google_maps_server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
    
            # Get tools
            tools = await load_mcp_tools(session)
    
            # Create and run the agent
            agent = create_react_agent(llm, tools)
            result = await agent.ainvoke(state)
            goto = get_next_node(result["messages"][-1], "map_plotter")
            return Command(
                update={
                    # share internal message history of chart agent with other agents
                    "messages": result["messages"][-1].content,
                },
                goto=goto,
            )

def plot_node(state: MessagesState) -> Command[Literal[END]]:
    output_notebook()
    bokeh_width, bokeh_height = 800,600
    map_type = "roadmap" #or satellite
    zoom = 10
    
    # Find latitude
    text = f"+++OutputFormat(format=JSON)\n\nExtract keys and values for latitude and longitude from the following text: {state['messages'][-1].content}"
    messages = [
        SystemMessage(content="You are a reliable assistant capable of providing truthful and accurate answers to questions."), 
        HumanMessage(content=text)
    ]
    print(f"In maps plotter: {text}")
    # Extract latitude and longitude
    data = json.loads(re.sub(r'```json\n|\n```', '', llm(messages).content))
    # Extract latitude and longitude
    print(f"-----Data: {data}")

    try: 
        if "latitude" in data or "Latitude" in data: 
            lat_match = data["latitude"] or data["Latitude"]
            lng_match = data["longitude"] or data["Longitude"]
        else:
            data_2 = list(data.keys())[0]
            lat_match = data["latitude"] or data["Latitude"]
            lng_match = data["longitude"] or data["Longitude"]
        
        if lat_match and lng_match:
            lat = float(lat_match)
            lng = float(lng_match)
            #lng = float(lng_match.group(1))
        else:
            lat = 0.00
            lng = 0.00
    except:
        lat = 0.00
        lng = 0.00

    print(f"lat and lng are: {lat} and {lng}")
    gmap_options = GMapOptions(lat=lat, lng=lng, map_type=map_type, zoom=zoom)
    p = gmap(GMAPS_API_KEY, gmap_options, title='MCP Maps', width=bokeh_width, height=bokeh_height)
    show(p)
    export_file = os.path.join(config['internal_file_path'], 'map.png')
    print(f'exporting to {export_file}')
    export_png(p, filename=export_file)

    goto = get_next_node(state["messages"][-1], END)
    # wrap in a human message, as not all providers allow
    # AI message at the last position of the input messages list
    state["messages"][-1] = HumanMessage(
        content=f"Map plot for {text}", name="map_plotter"
    )
    return Command(
        update={
            # share internal message history of chart agent with other agents
            "messages": '' #state["messages"][-1].content,
        },
        goto=goto,
    )

async def perplexity_node(state: MessagesState) -> Command[Literal[END]]:
    async with stdio_client(perplexity_server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            print(f"In mcp perplexity: {state}")
            # Get tools
            tools = await load_mcp_tools(session)
    
            # Create and run the agent
            agent = create_react_agent(llm, tools)
            result = await agent.ainvoke(state) #({"messages": "what's (3 + 5) x 12?"})
            goto = get_next_node(result["messages"][-1], END)
            return Command(
                update={
                    # share internal message history of chart agent with other agents
                    "messages": result["messages"][-1].content,
                },
                goto=goto,
            )

def get_next_node(last_message: BaseMessage, goto: str):
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return END
    return goto


# --------------------------------------------------------------------------------------------
# Deployment Nodes ---------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------


def invoke_model(state: MultiAgentState):
    pass

def response_reflection(state: MultiAgentState):
    # Answers:
    # GOOD - send to client
    # BAD - try again
    pass

def choose_action(state: MultiAgentState):
    # Answers:
    # MCP - send to client
    # CACHE - try again
    pass

def mcp_access(state: MultiAgentState):
    pass

def cache_access(state: MultiAgentState):
    # GOOD - send to client
    # BAD - try model
    pass
