from langgraph.graph import START, StateGraph, MessagesState, END
from langgraph.pregel import RetryPolicy
from asyncio import gather

from agents import *
from tools import *
from mcp_server import *
from agent_protocol import MultiAgentState

class Orchestration():
    def __init__(self):
        self.orch = StateGraph(MultiAgentState)

        # Nodes ------------------------
        # Rewrite node first aid
        self.orch.add_node("rewrite_node", rewrite_node)
        # Tools subgraph first aid
        self.orch.add_node("tools_subgraph", ToolsSubgraph)
        # Tools aggregator node first aid
        self.orch.add_node("answer_with_tools_node", answer_with_tools_node)
        # Quality assurance node first aid
        self.orch.add_node("judge_node", judge_node)
        
        # Edges ------------------------
        self.orch.set_entry_point("rewrite_node")
        self.orch.add_edge("rewrite_node", 'tools_subgraph')
        self.orch.add_edge("tools_subgraph", "answer_with_tools_node")
        self.orch.add_edge("answer_with_tools_node", 'judge_node')
        self.orch.add_conditional_edges(
            "judge_node",
            judge_decision,
            {
                'GOOD': END,
                'BAD': 'tools_subgraph'
            }
        )

        self.graph = self.orch.compile()

    async def invoke(self, prompt: str):
        events = self.graph.astream(
            {"question": prompt},
            # Maximum number of steps to take in the graph
            {"recursion_limit": 30},
        )
    
        out = []
        async for s in events:
            out.append(s)
        yield out


async def ToolsSubgraph(state: MultiAgentState):
    tools = {}

    # Tools ------------------------
    # PubMed access node
    tools["pubmed_node"] = pubmed_node
    # MedRXIV access node
    tools["medrxiv_node"] = medrxiv_node
    # Web Search node
    tools["web_search_node"] = web_search_node
    
    # Invoke tools concurrently ----
    state = state | tools_router_node(state)
    requested_tool_names = state['tools_requested']
    tools_to_use = [tools[tool](state) for tool in requested_tool_names]

    results = await gather(*tools_to_use)

    return state


"""
class TestingOrchestration():
    def __init__(self):
        workflow = StateGraph(MessagesState)
        workflow.add_node("rewrite_client", rewrite_node)
        workflow.add_node("mcp_client", mcp_nodes, retry=RetryPolicy(max_attempts=2))
        workflow.add_node("perplexity_client", perplexity_node)
        workflow.add_edge(START, "rewrite_client")
        workflow.add_conditional_edges("rewrite_client", router_dispatcher)
        
        sub2_workflow = StateGraph(MessagesState)
        sub2_workflow.add_node("gmaps_client", gmaps_node)
        sub2_workflow.add_node("map_plotter", plot_node)
        sub2_workflow.add_edge(START, "gmaps_client")
        subgraph_maps = sub2_workflow.compile()
        
        workflow.add_node("sub_graph_maps", subgraph_maps)
        workflow.add_edge("sub_graph_maps", END)
        
            
        '''
        # Add edges
        workflow.add_conditional_edges(
            START,
            rewrite_client,
            router_dispatcher,
             {
                "stdio_client": "stdio_client",
                "researcher": "researcher"
             },
        )
        '''
        
        self.graph = workflow.compile()

    async def invoke(self, prompt: str):
        events = self.graph.astream(
            {"messages": prompt},
            # Maximum number of steps to take in the graph
            {"recursion_limit": 15},
        )
    
        out = []
        async for s in events:
            out.append(s)
        yield out
"""