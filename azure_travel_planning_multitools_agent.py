import os
import json

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
from pathlib import Path
from rich import print as pprint

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

required_settings = {
    "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
    "AZURE_OPENAI_API_VERSION": AZURE_OPENAI_API_VERSION,
    "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
}

missing_settings = [name for name, value in required_settings.items() if not value]

if missing_settings:
    raise ValueError(f"Missing required settings: {", ".join(missing_settings)}")

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    deployment_name=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
)

@tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""

    weather_data = {
        "Tokyo": {
            "condition": "Cloudy",
            "temperature": "15°C"
        },
        "Osaka": {
            "condition": "sunny",
            "temperature": "20°C"
        },
        "Taipei": {
            "condition": "Rainy",
            "temperature": "25°C"
        }
    }

    return weather_data.get(city, {"condition": "Unknown", "temperature": None})

@tool
def calculate_trip_budget(days: int, hotel_per_night: int, daily_expense: int) -> str:
    """Calculate the total trip budget based on the number of days, hotel cost per night, and daily expenses."""
    hotel_cost = days * hotel_per_night
    expense_cost = days * daily_expense
    total_cost = hotel_cost + expense_cost

    return json.dumps(
        {
            "days": days,
            "hotel_cost_twd": hotel_cost,
            "daily_expense_twd": expense_cost,
            "total_cost_twd": total_cost
        }
    )

@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert currency from one type to another."""

    exchange_rates = {
        ("TWD", "JPY"): 4.8,
        ("JPY", "TWD"): 0.208
    }

    key = (from_currency.upper(), to_currency.upper())

    rate = exchange_rates.get(key)

    if rate is None:
        return json.dumps({"error": f"Unsupported conversion: {from_currency} to {to_currency}"}, ensure_ascii=False, indent=4)

    converted_amount = amount * rate

    return json.dumps({"amount": amount, "from_currency": from_currency, "to_currency": to_currency, "exchange_rate": rate, "converted_amount": round(converted_amount, 2)}, ensure_ascii=False, indent=4)

agent = create_agent(
    model=llm,
    tools=[get_weather, calculate_trip_budget, convert_currency],
    system_prompt="""
    You are a helpful travel planning assistant.
    Use get_weather when the user asks about the weather.
    Use calculate_trip_budget when the user asks about travel expenses or budget.
    Use convert_currency when the user asks about currency conversion.
    A single task may require multiple tools .
    Use all necessary tools before answering.
    Clearly explain which information came from each tool.
    """
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "I plan to visit Tokyo for 5 days. Please check the weather, calculate the budget with a hotel cost of 3,000 TWD per night and daily expenses of 1,500 TWD, and convert the total budget into JPY."
            }
        ]
    }
)

pprint(result)
# agent_result_json = json.dumps(result, ensure_ascii=False, indent=4)
# file_path = Path.cwd() / "agent_result.json"
# with open(file_path, "w", ensure_ascii=False) as f:
#     f.write(agent_result_json)

for index, message in enumerate(result["messages"]):
    print(f"\nMessage {index + 1}")
    print("Type:", type(message).__name__)

    if message.content:
        print("Content:")
        print(message.content)

    tool_calls = getattr(message, "tool_calls", None)

    if tool_calls:
        print("Tool Calls:")

        for tool_call in tool_calls:
            print("Tool name:", tool_call["name"])
            print("Tool arguments:", tool_call["args"])

print("\n======= Final Answer ======")
print(result["messages"][-1].content)


# print(get_weather.get_input_schema().model_json_schema())
# print(calculate_trip_budget.get_input_schema().model_json_schema())
# print(convert_currency.get_input_schema().model_json_schema())