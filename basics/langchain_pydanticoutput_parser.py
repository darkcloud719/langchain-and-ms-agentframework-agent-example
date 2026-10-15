import os

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from rich import print as pprint


# ============================================================
# Load Environment Variables
# ============================================================

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv(
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"
)


required_settings = {
    "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
    "AZURE_OPENAI_API_VERSION": AZURE_OPENAI_API_VERSION,
    "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
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
# LLM
# ============================================================

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    model=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
)


# ============================================================
# Pydantic Schema
# ============================================================

class TranslationResult(BaseModel):

    translated_text: str = Field(
        description="The English translation of the Mandarin text."
    )

    tone: str = Field(
        description="The tone or style of the translation."
    )


# ============================================================
# Pydantic Output Parser
# ============================================================

parser = PydanticOutputParser(
    pydantic_object=TranslationResult
)


# ============================================================
# Prompt
# ============================================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a Mandarin-to-English translator.

Translate the user's text into English.

{format_instructions}
"""
    ),
    (
        "user",
        "{text}"
    )
])


# Insert parser formatting instructions into prompt
prompt = prompt.partial(
    format_instructions=parser.get_format_instructions()
)


# ============================================================
# LCEL Chain
# ============================================================

chain = (
    prompt
    | llm
    | parser
)


# ============================================================
# Invoke
# ============================================================

result = chain.invoke({ 
    "text": "我是一名軟體工程師"
})


# ============================================================
# Result
# ============================================================

pprint(result)

print(type(result))

print(result.translated_text)

print(result.tone)

print(parser.get_format_instructions())