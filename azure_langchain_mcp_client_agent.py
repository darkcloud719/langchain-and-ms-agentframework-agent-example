import asyncio
import os 
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_ENDPOINT= os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

required_settings = {
    "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
    "AZURE_OPENAI_API_VERSION": AZURE_OPENAI_API_VERSION,
    "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
}

missing_settings = [key for key, value in required_settings.items() if not value]

if missing_settings:
    raise ValueError(f"Missing required settings: {', '.join(missing_settings)}")

async def main():

    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        deployment_name=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
    )

    mcp_config = {
        "mcpServers": {
            "my_server": {
                "url": "http://localhost:8000/mcp"
            }
        }
    }

    async with MCPAdapter(mcp_config) as adapter:

        tools = await adapter.list_tools()

        print("Available MCP tools:")
        for tool in tools:
            print(f"- {tool.name}")


        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="""
            You are a helpful assistant.

            Use MCP tools whenever they are useful.
            Clearly explain the result to the user.
            """
        )

        result = await agent.ainvoke({"messages": [{"role":"user", "content": "Use the MCP server to help me."}]})

        for index, message in enumerate(result["messages"], start=1):
            print(f"\nMessage {index}")
            print(f"Type:{type(message).__name__}")

            if message.content:
                print(f"Content: {message.content}")

            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                print("Tool calls:")

                for tool_call in tool_calls:
                    print(f"Tool name: {tool_call['name']}")
                    print(f"Tool arguments: {tool_call['args']}")

        print("\n=========== FINAL ANSWER =========")
        print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())

        



