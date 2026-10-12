import os

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_openai import AzureChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

required_settings = {
    "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
    "AZURE_OPENAI_API_VERSION": AZURE_OPENAI_API_VERSION,
    "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
    "TAVILY_API_KEY": TAVILY_API_KEY
}

missing_settings = [key for key, value in required_settings.items() if not value]

if missing_settings:
    raise ValueError(f"Missing required settings: {', '.join(missing_settings)}")


llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    deployment_name=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
)

tavily_search = TavilySearch(
    max_results=2,
    topic="finance",
    api_key=TAVILY_API_KEY
)

research_subagents = [
    {
        "name": "academic_researcher",
        "description": (
            "Finds academic studies and official reports about AI "
            "in education. Use for evidence, measured outcomes, and citations."
        ),
        "system_prompt": """
You are an academic research assistant.
Search for studies and official reports about AI in education.
Prioritize primary research and credible institutions.
Return 3 to 5 key findings with source URLs.
Clearly distinguish evidence from opinion.
""",
        "tools": [tavily_search],
        "model": llm,
    },
    {
        "name": "news_researcher",
        "description": (
            "Finds recent news and current developments about AI "
            "in education."
        ),
        "system_prompt": """
You are a news researcher.
Search for recent developments about AI in education.
Return the publication date, key point, and source URL for each item.
Avoid repeating the same story.
""",
        "tools": [tavily_search],
        "model": llm,
    },
    {
        "name": "risk_analyst",
        "description": (
            "Researches risks of AI in education, including privacy, "
            "accuracy, bias, and academic integrity."
        ),
        "system_prompt": """
You are a careful risk analyst.
Research risks of using AI in education.
For each risk, provide a short explanation and source URL.
Do not present speculation as established fact.
""",
        "tools": [tavily_search],
        "model": llm,
    },
]

agent = create_deep_agent(
    model=llm,
    subagents=research_subagents,
    system_prompt="""
    You are the lead research coordinator.

    For broad questions about AI in education, delegate distinct research
    tasks to the relevant subagents:
    - academic_researcher for studies and official reports
    - news_researcher for recent developments
    - risk_analyst for risks

    After receiving their findings:
    1. Combine overlapping points.
    2. Clearly separate research evidence, recent news, and risks.
    3. Include source URLs.
    4. Mention disagreements or evidence gaps.
    5. Do not claim that the subagents' findings are automatically correct;
    review and synthesize them before answering.
    """
)

result = agent.invoke({"messages": [{"role":"user","content":"請整理 AI 在教育上的研究證據、近期發展與主要風險，並附上來源。"}]})

for index, message in enumerate(result["messages"], start=1):
    print(f"\nMessage {index}:")
    print(f"Type: {type(message).__name__}")
    print(f"Content: {message.content}")

    tool_calls = getattr(message, "tool_calls", None)

    if tool_calls:
        print("Tool Calls:")

        for tool_call in tool_calls:
            print(f"Tool name: {tool_call['name']}")
            print(f"Tool arguments: {tool_call['args']}")

    print("-"* 60)

last_message = result["messages"][-1]

print("\nFinal Answer:")
print(last_message.content)