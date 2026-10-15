import os, json, logging, sys, time
from pathlib import Path
from datetime import datetime, timezone
from rich import print as pprint
import uuid
import tiktoken
from pypdf import PdfReader
from random import randint
from pydantic import BaseModel, ValidationError, Field
from openai import AzureOpenAI, APITimeoutError, APIConnectionError, RateLimitError, InternalServerError
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.models import QueryType, QueryCaptionType, QueryCaptionResult, QueryAnswerType, QueryCaptionResult, QueryAnswerResult, VectorizedQuery, VectorizableTextQuery, VectorFilterMode
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchIndexerDataContainer,
    SearchIndexer,
    SimpleField,
    SearchFieldDataType,
    EntityRecognitionSkillV3,
    SentimentSkillV3,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    SearchIndexerSkillset,
    SearchableField,
    IndexingParameters,
    SearchIndexerDataSourceConnection,
    IndexingParametersConfiguration,
    IndexingSchedule,
    CorsOptions,
    SearchIndexer,
    FieldMapping,
    ScoringProfile,
    ComplexField,
    ImageAnalysisSkill,
    OcrSkill,
    VisualFeature,
    TextWeights,
    SearchField,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    WebApiSkill
)
from typing import List
from rich import print as pprint
from rich.logging import RichHandler
from rich.table import Table
from rich.console import Console
from pathlib import Path
from zoneinfo import ZoneInfo

load_dotenv()

console = Console()

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

console_handler = RichHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler("app.log", mode="a", encoding="utf-8")
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

AZURE_OPENAI_EMBEDDING_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
AZURE_OPENAI_EMBEDDING_API_KEY = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_SOURCE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_SOURCE_CONTAINER_NAME")
AZURE_STORAGE_PROCESSED_CONTAINER_NAME = os.getenv("AZURE_STORAGE_PROCESSED_CONTAINER_NAME")

AZURE_AI_SEARCH_ENDPOINT = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
AZURE_AI_SEARCH_API_KEY = os.getenv("AZURE_AI_SEARCH_API_KEY")
AZURE_AI_SEARCH_INDEX_NAME = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
AZURE_AI_SEARCH_VECTOR_INDEX_NAME = os.getenv("AZURE_AI_SEARCH_VECTOR_INDEX_NAME")
AZURE_AI_SEARCH_INDEXER_NAME = os.getenv("AZURE_AI_SEARCH_INDEXER_NAME")
AZURE_AI_SEARCH_DATASOURCE_NAME = os.getenv("AZURE_AI_SEARCH_DATASOURCE_NAME")
AZURE_AI_SEARCH_SKILLSET_NAME = os.getenv("AZURE_AI_SEARCH_SKILLSET_NAME")

CHUNK_SIZE = 450
CHUNK_OVERLAP = 150
TOKEN_ENCODING = "cl100k_base"

enc = tiktoken.get_encoding(TOKEN_ENCODING)


required_settings = {
    "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
    "AZURE_OPENAI_API_VERSION": AZURE_OPENAI_API_VERSION,
    "AZURE_OPENAI_ENDPOITN": AZURE_OPENAI_ENDPOINT,
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
    "AZURE_OPENAI_EMBEDDING_ENDPOITN": AZURE_OPENAI_EMBEDDING_ENDPOINT,
    "AZURE_OPENAI_EMBEDDING_API_KEY": AZURE_OPENAI_EMBEDDING_API_KEY,
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    "AZURE_STORAGE_CONNECTION_STRING": AZURE_STORAGE_CONNECTION_STRING,
    "AZURE_STORAGE_SOURCE_CONTAINER_NAME": AZURE_STORAGE_SOURCE_CONTAINER_NAME,
    "AZURE_STORAGE_PROCESSED_CONTAINER_NAME": AZURE_STORAGE_PROCESSED_CONTAINER_NAME,
    "AZURE_AI_SEARCH_ENDPOINT": AZURE_AI_SEARCH_ENDPOINT,
    "AZURE_AI_SEARCH_API_KEY": AZURE_AI_SEARCH_API_KEY,
    "AZURE_AI_SEARCH_INDEX_NAME": AZURE_AI_SEARCH_INDEX_NAME,
    "AZURE_AI_SEARCH_VECTOR_INDEX_NAME": AZURE_AI_SEARCH_VECTOR_INDEX_NAME,
    "AZURE_AI_SEARCH_INDEXER_NAME": AZURE_AI_SEARCH_INDEXER_NAME,
    "AZURE_AI_SEARCH_DATASOURCE_NAME": AZURE_AI_SEARCH_DATASOURCE_NAME,
    "AZURE_AI_SEARCH_SKILLSET_NAME": AZURE_AI_SEARCH_SKILLSET_NAME
}

missing_settings = [key for key, value in required_settings.items() if not value]

if missing_settings:
    logger.error(f"Missing required settings: {', '.join(missing_settings)}")
    pprint(f"[red]Missing required settings: {', '.join(missing_settings)}[/red]")
    raise ValueError(f"Missing required settings: {', '.join(missing_settings)}")

client_embedding = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)

client_chat = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)

service_endpoint = AZURE_AI_SEARCH_ENDPOINT
key = AZURE_AI_SEARCH_API_KEY
index_vector_name = "linebot-vector-index"


def _clear_documents_in_index():
    try:
        with SearchClient(service_endpoint, index_vector_name, AzureKeyCredential(key)) as search_client:
            batch_size = 1000
            total_deleted = 0

            while True:
                results = search_client.search(search_text="*", select=["id"], top=batch_size)
                documents = [{"id":result["id"]} for result in results]
                if not documents:
                    break

                delete_results = search_client.delete_documents(documents)
                success_count = sum(1 for result in delete_results if result.succeeded)
                total_deleted += success_count

                pprint(f"[yellow]Deleted {success_count} documents in this batch. Total deleted: {total_deleted}[/yellow]")

                time.sleep(1)

            pprint(f"[green]Index `{index_vector_name}` cleared successfully. Total deleted: {total_deleted}[/green]")
    except Exception as e:
        logger.error(f"Failed to clear index `{index_vector_name}`: {e}")
        pprint(f"[red]Failed to clear index `{index_vector_name}`: {e}[/red]")


def _read_pdf(pdf_path: str) -> str:
    """
    Read text from a PDF file.
    """

    pdf_path = Path.cwd() / pdf_path

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))

    pages = []

    for page_number, page in enumerate (reader.pages, start=1):
        text = page.extract_text()

        if not text:
            continue

        pages.append(text)

    full_text = "\n\n".join(pages)

    if not full_text.strip():
        raise ValueError(f"No text could be extracted from {pdf_path}. The PDF may be a scanned/image PDF and require OCR")

    logger.info(f"Read PDF `{pdf_path.name}` pages={len(reader.pages)}, characters={len(full_text)}")

    output_filename = Path.cwd() / "output.txt"

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(full_text)

    return full_text

def _chunk_document_object_by_tokens(object_dict: dict) -> list[object]:
    """
    Chunk text from a document into smaller pieces based on token count.
    """
    try:
        logger.info(f"[Document Chunking] Chunking document object: {object_dict.get('filename', 'unknown_document')}")
        text = object_dict.get("content", "")
        tokens = enc.encode(text)
        chunks_objects = []

        start = 0
        while start < len(tokens):
            end = start + CHUNK_SIZE
            chunk = enc.decode(tokens[start:end])
            object_chunk = object_dict.copy()
            object_chunk["content"] = chunk
            chunks_objects.append(object_chunk)
            start = end - CHUNK_OVERLAP if end - CHUNK_OVERLAP > start else end
        return chunks_objects
    except Exception as e:
        logger.error(f"Failed to chunk document object: {e}")
        pprint(f"[red]Failed to chunk document object: {e}[/red]")
        return []

def upload_documents_to_search(documents: List):
    try:
        logger.info(f"[Document Upload] Uploading filename `{documents[0].get('filename', '')}`, total documents: {len(documents)} to Azure AI Search Vector Index: `{index_vector_name}`")
        with SearchClient(service_endpoint, index_vector_name, AzureKeyCredential(key)) as search_client:
            result = search_client.upload_documents(documents=documents)

            success_count = sum(1 for r in result if r.succeeded)
            logger.info(f"[Document Upload] Successfully uploaded {success_count}/{len(documents)} documents.")
    except Exception as e:
        logger.exception(f"[Document Upload] Failed to upload documents: {e}")

def _process_document_to_search():
    """
    Process the document text and perform a search.
    """
    try:
        filename = Path.cwd() / "output.txt"
        if not filename.exists():
            raise FileNotFoundError(f"Output file not found: {filename}")
        with open(filename, "r", encoding="utf-8") as f:
            document_text = f.read()
            logger.info(f"[Document Processing] Read document text from `{filename.name}`")
            print(f"document_text:{document_text}")
            json_data = {
                "filetype": filename.suffix.lstrip("."),
                "timestamp": datetime.now().isoformat(),
                "filename": filename.name,
                "content": document_text
            }

            chunks_objects = _chunk_document_object_by_tokens(json_data)

            logger.info(f"[Document Processing] Created {len(chunks_objects)} chunks from document `{filename.name}`")

            documents = []

            for idx, chunk_object in enumerate(chunks_objects):
                content = chunk_object.get("content", "")

                if not content:
                    continue

                embedding = client_embedding.embeddings.create(
                    model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                    input=content
                )

                embedding_vector = embedding.data[0].embedding


                documents.append({
                    "id": str(uuid.uuid4()),
                    "filename": chunk_object.get("filename", ""),
                    "filetype": chunk_object.get("filetype", ""),
                    "chunk_id": idx,
                    "timestamp": datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M:%S"),
                    "content": content,
                    "contentVector": embedding_vector
                })

            if documents:

                logger.info(f"[Document Processing] Uploading filename={documents[0].get('filename', '')} with {len(documents)} chunks.")

                upload_documents_to_search(documents)

    except Exception as e:
        logger.exception(f"[Document Processing] Failed to process document: {e}")

def hybrid_search_documents():
    try:
        query = "scalable storage solutions"

        embedding = client_embedding.embeddings.create(
            input=query,
            model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
        ).data[0].embedding

        vector_query = VectorizedQuery(
            vector=embedding,
            # k=3,
            fields="contentVector"
        )

        with SearchClient(service_endpoint, index_name=index_vector_name, credential=AzureKeyCredential(key)) as search_client:
            results = search_client.search(
                query_type=QueryType.SIMPLE,
                search_text=query,
                vector_queries=[vector_query],
                # select=["title","content","category","titleVector","contentVector"],
                top=2
            )

            all_records = list(results)

            # print_search_results(all_records)

            pprint(all_records)

    except Exception as e:
        pprint(f"[red]Error performing hybrid search: {e}[/red]")
        logger.error(f"Error performing hybrid search: {e}")


if __name__ == "__main__":
    # _read_pdf("student_manual.pdf")
    # _process_document_to_search()
    hybrid_search_documents()





