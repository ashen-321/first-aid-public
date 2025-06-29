from agno.agent import Agent
from agno.models.openai import OpenAIChat  # could be OpenRouter as well
from openai import OpenAI
import os
from agno.tools.mcp import MCPTools
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from langchain_openai import ChatOpenAI
#from agno.models.openai import OpenAIChat
import asyncio
from textwrap import dedent
from agno.knowledge.url import UrlKnowledge
#from agno.models.openai import OpenAIChat
from agno.models.aws.bedrock import AwsBedrock
from agno.models.aws.claude import Claude
from agno.playground import Playground, serve_playground_app
from agno.storage.sqlite import SqliteStorage
from agno.team import Team
from agno.tools.tavily import TavilyTools
from agno.tools.knowledge import KnowledgeTools
from agno.tools.reasoning import ReasoningTools
from agno.tools.thinking import ThinkingTools
from agno.tools.yfinance import YFinanceTools
#from agno.vectordb.lancedb import LanceDb, SearchType
from agno.vectordb.chroma import ChromaDb
import configparser
from agno.models.vllm import vLLM
from agno.tools.mcp import MCPTools
from textwrap import dedent
from agno.embedder.openai import OpenAIEmbedder 
from agno.embedder.aws_bedrock import AwsBedrockEmbedder

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

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    agent_id="finance-agent",
    model=model_gemma,
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
            company_news=True,
        )
    ],
    instructions=["Always use tables to display data"],
    storage=SqliteStorage(
        table_name="finance_agent", db_file=agent_storage_file, auto_upgrade_schema=True
    ),
    add_history_to_messages=True,
    num_history_responses=5,
    add_datetime_to_instructions=True,
    markdown=True,
)

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

finance_agent_2 = Agent(
    name="Finance Agent2",
    role="Get financial data with tool specified by a MCP server",
    agent_id="finance-agent-2",
    model=model_qwen,
    tools=[
        sequential_thinking_mcp_tools,
        stream_mcp_arxiv,
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
            company_news=True,
        )
    ],
    instructions=dedent("""\
                ## Using the think tool
                Before taking any action or responding to the user after receiving tool results, use the think tool as a scratchpad to:
                - List the specific rules that apply to the current request
                - Check if all required information is collected
                - Verify that the planned action complies with all policies
                - Iterate over tool results for correctness

                ## Rules
                - Its expected that you will use the think tool generously to jot down thoughts and ideas.
                - Use tables where possible\
                """),
    storage=SqliteStorage(
        table_name="finance_agent", db_file=agent_storage_file, auto_upgrade_schema=True
    ),
    add_history_to_messages=True,
    num_history_responses=5,
    add_datetime_to_instructions=True,
    markdown=True,
)

thinking_tool_agent = Agent(
    name="Thinking Tool Agent",
    agent_id="thinking_tool_agent",
    model=model_gemma,
    tools=[ThinkingTools(add_instructions=True), YFinanceTools(enable_all=True)],
    instructions=dedent("""\
        You are a seasoned Wall Street analyst with deep expertise in market analysis! 📊

        Follow these steps for comprehensive financial analysis:
        1. Market Overview
           - Latest stock price
           - 52-week high and low
        2. Financial Deep Dive
           - Key metrics (P/E, Market Cap, EPS)
        3. Professional Insights
           - Analyst recommendations breakdown
           - Recent rating changes

        4. Market Context
           - Industry trends and positioning
           - Competitive analysis
           - Market sentiment indicators

        Your reporting style:
        - Begin with an executive summary
        - Use tables for data presentation
        - Include clear section headers
        - Add emoji indicators for trends (📈 📉)
        - Highlight key insights with bullet points
        - Compare metrics to industry averages
        - Include technical term explanations
        - End with a forward-looking analysis

        Risk Disclosure:
        - Always highlight potential risk factors
        - Note market uncertainties
        - Mention relevant regulatory concerns\
    """),
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

reasoning_finance_team = Team(
    name="Reasoning Finance Team",
    mode="coordinate",
    model=model_qwen,
    members=[
        web_agent,
        finance_agent_2,
        thinking_tool_agent,
        cot_agent,
        reasoning_tool_agent,
        reasoning_model_agent,
    ],
    #reasoning=True,  # Does not work with model_qwen??
    tools=[ReasoningTools(add_instructions=True)],
    # uncomment it to use knowledge tools
    # tools=[knowledge_tools],
    team_id="reasoning_finance_team",
    debug_mode=True,
    instructions=[
        "Only output the final answer, no other text.",
        "Use tables to display data",
    ],
    markdown=True,
    show_members_responses=True,
    enable_agentic_context=True,
    add_datetime_to_instructions=True,
    success_criteria="The team has successfully completed the task.",
    storage=SqliteStorage(
        table_name="reasoning_finance_team",
        db_file=agent_storage_file,
        auto_upgrade_schema=True,
    ),
)

if __name__ == "__main__":
    reasoning_finance_team.print_response("Based on President Trump's recent social media posts and annoucements, recommend an investment portfolio by calling out the companies and amount to invest for $100K with targetted return at 7.5% annual with risk tolerance at high level", stream=True)