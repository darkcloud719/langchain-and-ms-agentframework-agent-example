import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_openai import AzureChatOpenAI


load_dotenv()


async def main():

    # -------------------------
    # 1. 建立 Azure OpenAI Model
    # -------------------------

    model = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ[
            "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"
        ],
    )


    # -------------------------
    # 2. 設定 MCP Server
    # -------------------------

    mcp_config = {
        "mcpServers": {
            "my_server": {
                "url": "http://localhost:8000/mcp"
            }
        }
    }


    # -------------------------
    # 3. 連線 MCP Server
    # -------------------------

    async with MCPAdapter(mcp_config) as adapter:

        # 取得 MCP Server 提供的所有工具
        tools = await adapter.list_tools()

        print("Available MCP tools:")
        for tool in tools:
            print("-", tool.name)


        # -------------------------
        # 4. 建立 LangChain Agent
        # -------------------------

        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt="""
            You are a helpful assistant.

            Use MCP tools whenever they are useful.
            Clearly explain the result to the user.
            """
        )


        # -------------------------
        # 5. 執行 Agent
        # -------------------------

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Use the MCP server to help me."
                    }
                ]
            }
        )


        # -------------------------
        # 6. 查看完整訊息流程
        # -------------------------

        for index, message in enumerate(
            result["messages"],
            start=1
        ):
            print(f"\nMessage {index}")
            print("Type:", type(message).__name__)

            if message.content:
                print("Content:", message.content)

            tool_calls = getattr(
                message,
                "tool_calls",
                None
            )

            if tool_calls:
                print("Tool Calls:")

                for tool_call in tool_calls:
                    print(
                        "Tool name:",
                        tool_call["name"]
                    )
                    print(
                        "Arguments:",
                        tool_call["args"]
                    )


        print("\n===== FINAL ANSWER =====")
        print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())