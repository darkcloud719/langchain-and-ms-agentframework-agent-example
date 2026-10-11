import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI


load_dotenv()


# -------------------------
# 1. 建立 Azure OpenAI Model
# -------------------------

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    deployment_name=os.getenv(
        "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"
    )
)


# -------------------------
# 2. 定義 Tool Input Schema
# -------------------------

class WeatherInput(BaseModel):
    city: str = Field(
        description="The city to check the weather for"
    )


# -------------------------
# 3. 定義 Tool Output Schema
# -------------------------

class WeatherOutput(BaseModel):
    city: str
    condition: str
    temperature_c: int


# -------------------------
# 4. 建立 Weather Tool
# -------------------------

@tool(args_schema=WeatherInput)
def get_weather(city: str) -> str:
    """
    Get the current weather for a city.
    This is a demo tool with mock data.
    """

    weather_data = {
        "Tokyo": {
            "condition": "Cloudy",
            "temperature_c": 15
        },
        "Osaka": {
            "condition": "Sunny",
            "temperature_c": 20
        },
        "Taipei": {
            "condition": "Rainy",
            "temperature_c": 25
        }
    }

    data = weather_data.get(
        city,
        {
            "condition": "Unknown",
            "temperature_c": 0
        }
    )

    result = WeatherOutput(
        city=city,
        condition=data["condition"],
        temperature_c=data["temperature_c"]
    )

    # 將 Pydantic Model 轉成 JSON 字串
    return result.model_dump_json()


# -------------------------
# 5. 查看 Tool Schema
# -------------------------

print("Tool name:")
print(get_weather.name)

print("\nTool description:")
print(get_weather.description)

print("\nTool schema:")
print(get_weather.get_input_schema().model_json_schema())


# -------------------------
# 6. 建立 Agent
# -------------------------

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="""
    You are a helpful weather assistant.

    Use the get_weather tool when the user asks about weather.
    After receiving the tool result, explain the weather clearly.
    """
)


# -------------------------
# 7. 呼叫 Agent
# -------------------------

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in Tokyo?"
            }
        ]
    }
)


# -------------------------
# 8. 查看完整訊息流程
# -------------------------

for index, message in enumerate(result["messages"], start=1):
    print(f"\nMessage {index}")
    print("Type:", type(message).__name__)

    if message.content:
        print("Content:", message.content)

    if getattr(message, "tool_calls", None):
        print("Tool Calls:")

        for tool_call in message.tool_calls:
            print("Tool name:", tool_call["name"])
            print("Tool arguments:", tool_call["args"])


# -------------------------
# 9. 顯示最後答案
# -------------------------

print("\nFinal Answer:")
print(result["messages"][-1].content)