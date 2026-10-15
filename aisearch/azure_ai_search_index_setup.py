import os, json, logging, sys, time
from pathlib import Path
from datetime import datetime, timezone
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
index_name = AZURE_AI_SEARCH_INDEX_NAME
index_vector_name = AZURE_AI_SEARCH_VECTOR_INDEX_NAME
indexer_name = AZURE_AI_SEARCH_INDEXER_NAME
container_name = AZURE_STORAGE_SOURCE_CONTAINER_NAME
datasource_name = AZURE_AI_SEARCH_DATASOURCE_NAME
skillset_name = AZURE_AI_SEARCH_SKILLSET_NAME

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

def _delete_index():
    try:
        with SearchIndexClient(service_endpoint, AzureKeyCredential(key)) as search_index_client:
            result = search_index_client.delete_index(index_vector_name)
            logger.info(f"Index `{index_vector_name}` deleted successfully.")
            pprint(f"[green]Index `{index_vector_name}` deleted successfully[/green]")
    except Exception as e:
        logger.error(f"Failed to delete index `{index_vector_name}`: {e}")
        pprint(f"[red]Failed to delete index `{index_vector_name}`: {e}")

def _create_vector_index():
    try:
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True, sortable=True),
            SimpleField(name="filetype", type=SearchFieldDataType.String, filterable=True, sortable=True, facetable=True),
            SimpleField(name="chunk_id", type=SearchFieldDataType.Int32),
            SimpleField(name="url", type=SearchFieldDataType.String),
            SearchableField(name="timestamp", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SearchableField(name="filename", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchField(
                name="contentVector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="my-vector-search-profile"
            )
        ]

        vector_search = VectorSearch(
            profiles=[VectorSearchProfile(name="my-vector-search-profile", algorithm_configuration_name="my-hnsw-algorithm")],
            algorithms=[HnswAlgorithmConfiguration(name="my-hnsw-algorithm")]
        )

        scoring_profiles:List[ScoringProfile] = []
        scoring_profile = ScoringProfile(
            name="my-scoring-profile",
            text_weights=TextWeights(weights={"content":2.0, "filename": 1.5})
        )

        scoring_profiles.append(scoring_profile)
        cors_options = CorsOptions(allowed_origins=["*"], max_age_in_seconds=60)

        semantic_config = SemanticConfiguration(
            name="my-semantic-configuration",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="filename"),
                keywords_fields=[SemanticField(field_name="content")],
                content_fields=[SemanticField(field_name="content")]
            )
        )

        semantic_search = SemanticSearch(configurations=[semantic_config])

        with SearchIndexClient(service_endpoint, AzureKeyCredential(key)) as search_index_client:
            index = SearchIndex(
                name=index_vector_name,
                fields=fields,
                cors_options=cors_options,
                scoring_profiles=scoring_profiles,
                semantic_search=semantic_search,
                vector_search=vector_search
            )

            result = search_index_client.create_index(index)

            logger.info(f"Vector index `{index_vector_name}` created successfully.")
            pprint(f"[green]Vector index `{index_vector_name}` created successfully.[/green]")

    except Exception as e:
        logger.error(f"Failed to create vector index `{index_vector_name}`: {e}")
        pprint(f"[red]Failed to create vector index `{index_vector_name}`: {e}[/red]")

def _delete_datasource_connection():
    try:
        with SearchIndexClient(service_endpoint, AzureKeyCredential(key)) as index_client:
            result = index_client.delete_data_source_connection(datasource_name)
            logger.info(f"Datasource connection `{datasource_name}` deleted successfully.")
            pprint(f"[green]Datasource connection `{datasource_name}` deleted successfully.[/green]")
    except Exception as e:
        logger.error(f"Failed to delete datasource connection `{datasource_name}`: {e}")
        pprint(f"[red]Error deleting datasource connection `{datasource_name}`: {e}[/red]")

def _create_datasource_connection():
    try:
        container = SearchIndexerDataContainer(name=container_name)
        datasource_connection = SearchIndexerDataSourceConnection(
            name=datasource_name,
            type="azureblob",
            connection_string=AZURE_STORAGE_CONNECTION_STRING,
            container=container
        )

        with SearchIndexerClient(service_endpoint, AzureKeyCredential(key)) as indexer_client:
            result = indexer_client.create_data_source_connection(datasource_connection)
            logger.info(f"Datasource connection `{datasource_name}` created successfully.")
            pprint(f"[green]Datasource connection `{datasource_name}` created successfully.[/green]")
    except Exception as e:
        logger.error(f"Error creating datasource connection `{datasource_name}`: {e}")
        pprint(f"[red]Error creating datasource connection `{datasource_name}`: {e}[/red]")

def _delete_skillset():
    try:
        with SearchIndexerClient(service_endpoint, AzureKeyCredential(key)) as indexer_client:
            result = indexer_client.delete_skillset(skillset_name)
            logger.info(f"Deleted skillset `{skillset_name}` successfully.")
            pprint(f"[green]Deleted skillset `{skillset_name}` successfully.[/green]")
    except Exception as e:
        logger.error(f"Failed to delete skillset `{skillset_name}`: {e}")
        pprint(f"[red]Failed to delete skillset `{skillset_name}`: {e}[/red]")

def _create_indexer():
    try:
        configuration = IndexingParametersConfiguration(
            parsing_mode="default",
            query_timeout=None
        )

        parameters = IndexingParameters(configuration=configuration)

        indexer = SearchIndexer(
            name=indexer_name,
            data_source_name=datasource_name,
            target_index_name=index_name,
            skillset_name=skillset_name,
            parameters=parameters,
            field_mappings=[
                FieldMapping(source_field_name="metadata_storage_path", target_field_name="url"),
                FieldMapping(source_field_name="metadata_storage_name", target_field_name="filename"),
                FieldMapping(source_field_name="metadata_content_type", target_field_name="filetype")
            ],
            # output_field_mappings=[
            #     FieldMapping(source_field_name="/document/text", target_field_name="text")
            # ]
        )

        with SearchIndexerClient(service_endpoint, AzureKeyCredential(key)) as search_indexer_client:
            search_indexer_client.create_indexer(indexer)
            logger.info(f"Indexer `{indexer_name}` created successfully.")
            pprint(f"[green]Indexer `{indexer_name}` created successfully.[/green]")
    except Exception as e:
        logger.error(f"Error creating indexer `{indexer_name}`: {e}")
        pprint(f"[red]Error creating indexer `{indexer_name}`: {e}[/red]")

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
        logger.exception(f"[Document Chunking] Failed to chunk document object `{object_dict.get('filename', 'Unknown_document')}`: {e}")
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


if __name__ == "__main__":
    # _delete_index()
    # _create_vector_index()
    # _read_pdf("student_manual.pdf")
    _process_document_to_search()





