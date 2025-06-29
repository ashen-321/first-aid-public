from agno.agent import Agent
import os
from agno.tools.mcp import MCPTools
from mcp import StdioServerParameters
from textwrap import dedent
from agno.tools.mcp import MultiMCPTools
from agno.models.openai import OpenAIChat
from langchain_openai import ChatOpenAI
#from agno.models.openai import OpenAIChat
import asyncio
from agno.knowledge.url import UrlKnowledge
#from agno.models.openai import OpenAIChat
from agno.models.aws.bedrock import AwsBedrock
from agno.storage.sqlite import SqliteStorage
from agno.team import Team
from agno.tools.tavily import TavilyTools
from agno.tools.knowledge import KnowledgeTools
from agno.tools.reasoning import ReasoningTools
from agno.tools.thinking import ThinkingTools
#from agno.vectordb.lancedb import LanceDb, SearchType
from agno.vectordb.chroma import ChromaDb
import configparser
from agno.models.vllm import vLLM
from agno.models.openai import OpenAIChat  # could be OpenRouter as well
from openai import OpenAI
from agno.embedder.openai import OpenAIEmbedder 
from agno.embedder.aws_bedrock import AwsBedrockEmbedder


ai_thinking_prompt = """
        You are a seasoned AI expert specializing in analyzing research papers from arXiv and code repositories from GitHub! 📊

        Follow these steps for comprehensive academic and technical analysis:
        1. Research Evaluation
           - Paper methodology assessment
           - Technical innovation analysis
           - Implementation feasibility
           - Code quality assessment
        
        2. Deep Technical Dive
           - Key algorithms and architectures
           - Mathematical formulations
           - Technical limitations
           - Experimental design critique
        
        3. Professional Insights
           - Research impact and significance
           - Implementation challenges
           - Comparative analysis with SOTA
           - Potential applications
        
        4. Technical Context
           - Related research connections
           - Code architecture evaluation
           - Algorithm complexity analysis
           - Theoretical foundations
        
        Your analytical style:
        - Begin with an executive summary of key findings
        - Use code blocks for illustrating implementations
        - Include clear technical section headers
        - Add evaluation indicators for strengths/weaknesses (🔍 ⚠️)
        - Highlight technical insights with bullet points
        - Compare with established technical benchmarks
        - Include technical term explanations
        - End with future research directions
        
        Technical Disclosure:
        - Always highlight computational complexity concerns
        - Note implementation challenges
        - Mention relevant theoretical limitations
        - Address reproducibility considerations
    """


agno_docs = UrlKnowledge(
    urls=["https://www.paulgraham.com/read.html"],
    # Use LanceDB as the vector database and store embeddings in the `agno_docs` table
    #vector_db=LanceDb(
    #    uri="tmp/pgdb",
    #    table_name="agno_docs",
    #    search_type=SearchType.hybrid,
    #    #embedder=OpenAIEmbedder(id="text-embedding-3-small")
    #    embedder=AwsBedrockEmbedder(id="cohere.embed-english-v3")
    #    embedder=AwsBedrockEmbedder(id="amazon.titan-embed-text-v2:0")
    #),
    vector_db=ChromaDb(
        persistent_client=True,
        path="tmp/chroma_db",
        collection="agno_docs",
        reranker=None,
        embedder=OpenAIEmbedder(id="text-embedding-3-small")
    )
)

knowledge_tools = KnowledgeTools(
    knowledge=agno_docs,
    think=True,
    search=True,
    analyze=True,
    add_few_shot=True,
)

sequential_thinking_mcp_tools = MCPTools(
            command="npx -y @modelcontextprotocol/server-sequential-thinking"
        ) 
stream_mcp_arxiv = MCPTools(
            url="http://infs.cavatar.info:8084/mcp", transport="streamable-http", timeout_seconds=20
        ) 

server_params_github = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
    )
server_params_sthinking = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-sequential-thinking"],
    )

github_mcp_server_url = "http://agent.cavatar.info:8080/mcp"
arxiv_mcp_server_url = "http://infs.cavatar.info:8084/mcp"

model_id_c37 = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
model_id_h35 = 'us.anthropic.claude-3-5-haiku-20241022-v1:0'
model_id_c35 = "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
model_id_nova = 'us.amazon.nova-lite-v1:0'
model_id_ds = 'us.deepseek.r1-v1:0'

# AWS Credentials
def get_aws_credentials_from_file():
    try:
        # Path to the AWS credentials file
        aws_credentials_path = os.path.expanduser("~/.aws/credentials")
        
        # Create a ConfigParser object
        config = configparser.ConfigParser()
        
        # Read the credentials file
        config.read(aws_credentials_path)
        
        # Extract the access key ID and secret access key from the [default] profile
        if 'default' in config:
            aws_access_key_id = config['default'].get('aws_access_key_id')
            aws_secret_access_key = config['default'].get('aws_secret_access_key')
            return aws_access_key_id, aws_secret_access_key
        else:
            print("No [default] profile found in credentials file")
            return None, None
            
    except Exception as e:
        print(f"Error reading AWS credentials: {e}")
        return None, None

#os.environ["OPENAI_API_KEY"] = openai_api_key = os.getenv("bedrock_api_token")
#os.environ["OPENAI_BASE_URL"] = openai_base_url = os.getenv("bedrock_api_url")
os.environ["TAVILY_API_KEY"] = tavily_api_key = os.getenv("tavily_api_token")
os.environ["AWS_ACCESS_KEY_ID"], os.environ["AWS_SECRET_ACCESS_KEY"] = get_aws_credentials_from_file()

# Standard LLM model (e.g., GPT-4 via OpenRouter/OpenAI)
#model = OpenAIChat(id=model_id_nova)  # assumes API key set via env
#model = Claude(id=model_id_c37, aws_region="us-west-2")  # assumes API key set via env
model_nova = AwsBedrock(id=model_id_nova)
model_c37 = AwsBedrock(id=model_id_c37)
model_h35 = AwsBedrock(id=model_id_h35)
model_ds = AwsBedrock(id=model_id_ds)
model_qwen = vLLM(base_url="http://agent.cavatar.info:8081/v1", api_key="EMPTY", id="Qwen/Qwen3-30B-A3B",temperature=0.2, top_p=0.90, presence_penalty=1.45)
model_gemma = vLLM(base_url="http://infs.cavatar.info:8081/v1", api_key="EMPTY", id="alfredcs/gemma-3-27b-grpo-med-merged",temperature=0.2, top_p=0.90, presence_penalty=1.45)


model_openai_br = ChatOpenAI(
    model=model_id_c37,
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=5,
    #api_key=os.getenv("bedrock_api_token"),  # if you prefer to pass api key in directly instaed of using env vars
    #base_url=os.getenv("bedrock_api_url"),
    # organization="...",
    # other params...
)

model_openai = OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("openai_api_token"), base_url="https://api.openai.com/v1/")

agent_storage_file: str = "tmp/agents.db"
image_agent_storage_file: str = "tmp/image_agent.db"

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"


cot_agent = Agent(
    name="Chain-of-Thought Agent",
    role="Answer basic questions",
    agent_id="cot-agent",
    model=model_ds,
    storage=SqliteStorage(
        table_name="cot_agent", db_file=agent_storage_file, auto_upgrade_schema=True
    ),
    add_history_to_messages=True,
    num_history_responses=3,
    add_datetime_to_instructions=True,
    markdown=True,
    reasoning=True,
)

reasoning_model_agent = Agent(
    name="Reasoning Model Agent",
    role="Reasoning about Math",
    agent_id="reasoning-model-agent",
    model=model_nova,
    reasoning_model=model_ds,
    instructions=["You are a reasoning agent that can reason about math."],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
    storage=SqliteStorage(
        table_name="reasoning_model_agent",
        db_file=agent_storage_file,
        auto_upgrade_schema=True,
    ),
)

reasoning_tool_agent = Agent(
    name="Reasoning Tool Agent",
    role="Answer basic questions",
    agent_id="reasoning-tool-agent",
    model=model_nova,
    storage=SqliteStorage(
        table_name="reasoning_tool_agent",
        db_file=agent_storage_file,
        auto_upgrade_schema=True,
    ),
    add_history_to_messages=True,
    num_history_responses=3,
    add_datetime_to_instructions=True,
    markdown=True,
    tools=[ReasoningTools()],
)

web_agent = Agent(
    name="Web Search Agent",
    role="Handle web search requests",
    model=model_gemma,
    agent_id="web_agent",
    tools=[TavilyTools()],
    instructions="Always include sources",
    add_datetime_to_instructions=True,
    storage=SqliteStorage(
        table_name="web_agent",
        db_file=agent_storage_file,
        auto_upgrade_schema=True,
    ),
    stream=True,
    stream_intermediate_steps=True,
)


thinking_tool_agent = Agent(
    name="Thinking Tool Agent",
    agent_id="thinking_tool_agent",
    model=model_gemma,
    tools=[ThinkingTools(add_instructions=True)],
    instructions=dedent(ai_thinking_prompt),
    add_datetime_to_instructions=True,
    show_tool_calls=True,
    markdown=True,
    stream_intermediate_steps=True,
    storage=SqliteStorage(
        table_name="thinking_tool_agent",
        db_file=agent_storage_file,
        auto_upgrade_schema=True,
    ),
)

knowledge_agent = Agent(
    agent_id="knowledge_agent",
    name="Knowledge Agent",
    model=model_ds,
    tools=[knowledge_tools],
    show_tool_calls=True,
    markdown=True,
    storage=SqliteStorage(
        table_name="knowledge_agent",
        db_file=agent_storage_file,
        auto_upgrade_schema=True,
    ),
)


async def call_mcp(prompt:str):
    with (    
        MCPTools(url=github_mcp_server_url, transport="streamable-http") as stream_github_mcp_tools,
        MCPTools(url=arxiv_mcp_server_url, transport="streamable-http") as stream_arxiv_mcp_tools,
        MCPTools(server_params=server_params_github ) as local_github_mcp_tools,
        MCPTools(server_params=server_params_sthinking ) as local_sthinking_mcp_tools,
    ):
            ai_agent = Agent(
                model=model_qwen,  #model_qwen works, so does model_gemma but not model_nova nor model_openai_br
                tools=[local_github_mcp_tools, stream_github_mcp_tools, stream_arxiv_mcp_tools, local_sthinking_mcp_tools],
                instructions=dedent("""\
                    You are a seasoned AI assistant. Help users explore arxiv ai and cs publications and github repositories and their activity.
        
                    - Use headings to organize your responses
                    - Be concise and focus on relevant information\
                """),
                markdown=True,
                show_tool_calls=True,
                debug_mode=False,
            )


            reasoning_ai_team = Team(
                name="Reasoning ai Team",
                mode="coordinate",
                model=model_qwen,  # Model_qwen without reasoning works
                members=[
                    web_agent,
                    #cot_agent,
                    ai_agent,
                    reasoning_tool_agent,
                    reasoning_model_agent,
                    thinking_tool_agent
                ],
                #reasoning=True, # Works with model_openai
                tools=[ReasoningTools(add_instructions=True)],
                # uncomment it to use knowledge tools
                # tools=[knowledge_tools],
                team_id="reasoning_ai_team",
                debug_mode=True,
                instructions=[
                    "Only output the trusted answers from github or search.",
                    "Use tables to display data and chart display architecture floes",
                ],
                markdown=True,
                show_members_responses=True,
                enable_agentic_context=True,
                add_datetime_to_instructions=True,
                success_criteria="The team has successfully completed the task.",
                storage=SqliteStorage(
                    table_name="reasoning_ai_team",
                    db_file=agent_storage_file,
                    auto_upgrade_schema=True,
                ),
            )
        
            await reasoning_ai_team.aprint_response(prompt, stream=True, stream_intermediate_steps=True)

if __name__ == "__main__":
    prompt = "Tell me about multi-agentic reasoning based on the Github repos https://github.com/agno-agi/agno and https://github.com/langchain-ai/langgraph. You can analyze the entire repos for detail information."
    messages=[
            {"role": "system", "content": "You are a helpful assistant. Please answer the user question accurately and truthfully. Also please make sure to think carefully before answering"},
            {"role": "user", "content": prompt},
    ],
    #loop = asyncio.get_event_loop()
    #loop.run_until_complete(call_mcp(prompt))
    asyncio.run(call_mcp(prompt))