import os

from dotenv import load_dotenv
from rich import print as pprint

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent


# ============================================================
# Load Environment Variables
# ============================================================

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv(
    "AZURE_OPENAI_API_KEY"
)

AZURE_OPENAI_API_VERSION = os.getenv(
    "AZURE_OPENAI_API_VERSION"
)

AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT"
)

AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv(
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"
)


required_settings = {
    "AZURE_OPENAI_API_KEY":
        AZURE_OPENAI_API_KEY,

    "AZURE_OPENAI_API_VERSION":
        AZURE_OPENAI_API_VERSION,

    "AZURE_OPENAI_ENDPOINT":
        AZURE_OPENAI_ENDPOINT,

    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME":
        AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
}


missing_settings = [
    key
    for key, value in required_settings.items()
    if not value
]


if missing_settings:
    raise ValueError(
        f"Missing required settings: "
        f"{', '.join(missing_settings)}"
    )


# ============================================================
# Create LLM
# ============================================================

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
    temperature=0
)


# ============================================================
# Tool 1 — Weather
# ============================================================

@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a specified city.
    """

    weather_data = {
        "New York": "Sunny, 25°C",
        "Los Angeles": "Cloudy, 22°C",
        "Chicago": "Rainy, 18°C",
        "San Francisco": "Foggy, 15°C"
    }

    return weather_data.get(
        city,
        "Weather data not available for this city."
    )


# ============================================================
# Tool 2 — Calculator
# ============================================================

@tool
def calculate(
    operation: str,
    number1: float,
    number2: float
) -> float:
    """
    Perform a basic mathematical calculation.

    Supported operations:
    add, subtract, multiply, divide.
    """

    if operation == "add":
        return number1 + number2

    elif operation == "subtract":
        return number1 - number2

    elif operation == "multiply":
        return number1 * number2

    elif operation == "divide":

        if number2 == 0:
            raise ValueError(
                "Cannot divide by zero."
            )

        return number1 / number2

    else:
        raise ValueError(
            f"Unsupported operation: {operation}"
        )


# ============================================================
# Tool 3 — Translation
# ============================================================

@tool
def translate_to_english(
    text: str
) -> str:
    """
    Translate Mandarin Chinese text into English.

    This demo returns predefined translations
    for educational purposes.
    """

    translation_data = {
        "今天天氣很好":
            "The weather is nice today.",

        "我今天很開心":
            "I am very happy today."
    }

    return translation_data.get(
        text,
        "Translation not available for this text."
    )


# ============================================================
# Tools
# ============================================================

tools = [
    get_weather,
    calculate,
    translate_to_english
]


# ============================================================
# Create Agent
# ============================================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a helpful assistant. "
        "Use tools when they are useful for answering the user."
    )
)


# ============================================================
# Invoke Agent
# ============================================================

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is 25 multiplied by 8?"
        }
    ]
})


# ============================================================
# Show Full Agent State
# ============================================================

pprint(
    result
)


# ============================================================
# Final Answer
# ============================================================

final_message = result["messages"][-1]

print(
    f"\nFinal Answer: {final_message.content}"
)