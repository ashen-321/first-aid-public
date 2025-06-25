from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import MessagesState, END
from langgraph.types import Command
from langchain_mcp_adapters.tools import load_mcp_tools
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
from openai import OpenAI

from tools import openai_url_invoke
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
# Deployment Nodes ---------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------


def rewrite_node(state: MultiAgentState):
    print(f'\n--- Rewriting question ---\n')
    
    llm = ChatOpenAI(
        model=config['model_id'],
        temperature=1.0,
        max_tokens=None,
        timeout=None,
        max_retries=5,
    )

    messages = [
        ("system", "You are a seasoned medical expert. Your role is to rewrite the user's query to include more relevant information to prompt other large language models to pick up on the nuances of the query and provide a more comprehensive answer."),
        ("human", state['question']),
    ]

    rewritten_question = llm.invoke(messages).content
    
    return {'rewritten_question': rewritten_question}


def tools_router_node(state: MultiAgentState):
    print(f'\n--- Selecting tools ---\n')
    
    tool_selection = ["pubmed_node", "medrxiv_node", "web_search_node"]

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


def answer_with_tools_node(state: MultiAgentState):
    print(f'\n--- Summarizing retrieved data ---\n')

    llm = ChatOpenAI(
        model=config['model_id'],
        temperature=0.5,
        max_tokens=None,
        timeout=None,
        max_retries=5,
    )

    messages = [
        ("system", f"You are a seasoned medical expert. Your role is to incorporate the provided information in your answer to increase its accuracy and comprehensiveness to the user's query:\n\n{state['external_summaries']}"),
        ("human", state['rewritten_question']),
    ]

    answer_with_tools = llm.invoke(messages).content
    
    return {'answer': answer_with_tools}


def judge_node(state: MultiAgentState):
    print(f'\n--- Judging response ---\n')
    # Answers:
    # GOOD - send to client
    # BAD - try again
    
    decision, _ = openai_url_invoke("alfredcs/gemma-3-27b-grpo-med-merged", state['answer'], "http://infs.cavatar.info:8081/v1").strip().upper()
    
    return {'qa_assessment': decision}


def judge_decision(state: MultiAgentState):
    print(f'\n--- Routing judge decision ---\n')
    
    return state['qa_assessment']
