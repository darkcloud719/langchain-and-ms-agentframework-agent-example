import os

from dotenv import load_dotenv
from rich import print as pprint

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    ToolMessage
)


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
    model=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
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
        "London": "Cloudy, 18°C",
        "Chicago": "Windy, 20°C",
        "Taipei": "Rainy, 22°C"
    }

    return weather_data.get(
        city,
        "Weather data not available."
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
    add, subtract, multiply, divide
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

    This demo returns a predefined translation
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
        "Translation not found."
    )


# ============================================================
# Tools
# ============================================================

tools = [
    get_weather,
    calculate,
    translate_to_english
]


# Create a lookup table
tools_by_name = {
    tool.name: tool
    for tool in tools
}


# ============================================================
# Bind Tools to LLM
# ============================================================

llm_with_tools = llm.bind_tools(
    tools
)


# ============================================================
# User Question
# ============================================================

question = (
    "What is 25 multiplied by 8?"
)


messages = [
    HumanMessage(
        content=question
    )
]


# ============================================================
# Step 1 — Let LLM Choose a Tool
# ============================================================

response = llm_with_tools.invoke(
    messages
)


print(
    "\n========== LLM Tool Decision ==========\n"
)

pprint(
    response.tool_calls
)


# Important:
# The assistant tool-call message must also be
# added to the conversation history.

messages.append(
    response
)


# ============================================================
# Step 2 — Execute Tools
# ============================================================

if response.tool_calls:

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]

        tool_args = tool_call["args"]

        tool_call_id = tool_call["id"]


        print(
            f"\nExecuting Tool: {tool_name}"
        )

        pprint(
            tool_args
        )


        # Find the actual LangChain Tool
        selected_tool = tools_by_name[
            tool_name
        ]


        # ================================================
        # ACTUAL TOOL EXECUTION
        # ================================================

        tool_result = selected_tool.invoke(
            tool_args
        )


        print(
            "\nTool Result:"
        )

        pprint(
            tool_result
        )


        # ================================================
        # Send Tool Result Back to LLM
        # ================================================

        messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call_id
            )
        )


# ============================================================
# Step 3 — Ask LLM for Final Answer
# ============================================================

final_response = llm_with_tools.invoke(
    messages
)


print(
    "\n========== Final Answer ==========\n"
)

print(
    final_response.content
)