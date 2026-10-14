import asyncio, os, json, logging, sys
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

AZURE_FUNCTION_APP_FOR_DOCUMENT_CRACKING_URL = os.getenv("AZURE_FUNCTION_APP_FOR_DOCUMENT_CRACKING_URL")
AZURE_FUNCTION_APP_FOR_EXTRACT_DOCUMENT_IMAGES_URL = os.getenv("AZURE_FUNCTION_APP_FOR_EXTRACT_DOCUMENT_IMAGES_URL")

required_settings = {
    "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
    "AZURE_OPENAI_API_VERSION": AZURE_OPENAI_API_VERSION,
    "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
    "AZURE_OPENAI_EMBEDDING_ENDPOINT": AZURE_OPENAI_EMBEDDING_ENDPOINT,
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
    "AZURE_AI_SEARCH_SKILLSET_NAME": AZURE_AI_SEARCH_SKILLSET_NAME,
    "AZURE_FUNCTION_APP_FOR_DOCUMENT_CRACKING_URL": AZURE_FUNCTION_APP_FOR_DOCUMENT_CRACKING_URL,
    "AZURE_FUNCTION_APP_FOR_EXTRACT_DOCUMENT_IMAGES_URL": AZURE_FUNCTION_APP_FOR_EXTRACT_DOCUMENT_IMAGES_URL
}

missing_settings = [key for key, value in required_settings.items() if not value]

if missing_settings:
    logger.error(f"Missing required settings: {', '.join(missing_settings)}")
    raise ValueError(f"Missing required settings: {', '.join(missing_settings)}")

client_embedding = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
    api_key=AZURE_OPENAI_EMBEDDING_API_KEY,
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

def print_search_result(results, top_k, query):
    table = Table(title=f"Top {top_k} SearchResults for Query: `{query}`")
    table.add_column("Id", style="cyan", no_wrap=True)
    table.add_column("Title", style="magenta")
    table.add_column("Category", style="green")
    table.add_column("Content", style="white")
    table.add_column("TitleVector", style="yellow")
    table.add_column("ContentVector", style="blue")

    for r in results:
        table.add_row(
            str(r.get("id", "")),
            r.get("title", ""),
            r.get("category", ""),
            r.get("content", "")[:100] + "..." if r.get("content", "") else "",
            str(r.get("titleVector", ""))[:10] + "..." if r.get("titleVector", "") else "",
            str(r.get("contentVector", ""))[:10] + "..." if r.get("contentVector", "") else ""
        )

    console.print(table)

def _delete_index():
    try:
        with SearchIndexClient(service_endpoint, AzureKeyCredential(key)) as search_index_client:
            result = search_index_client.delete_index(index_name)
            logger.info(f"Index `{index_name}` deleted successfully.")
            pprint(f"[yellow]Index `{index_name}` deleted successfully.[/yellow]")
    except Exception as e:
        logger.error(f"Failed to delete index `{index_name}`: {e}")
        pprint(f"[red]Failed to delete index `{index_name}`: {e}[/red]")

def _create_index():
    try:
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True, sortable=True),
            SimpleField(name="filetype", type=SearchFieldDataType.String, filtrable=True, sortable=True, facetable=True),
            SearchableField(name="url", type=SearchFieldDataType.String, filtrable=True, sortable=True),
            SearchableField(name="filename", type=SearchFieldDataType.String, filtrable=True, sortable=True),
            SimpleField(name="image_count", type=SearchFieldDataType.Int32),
            SearchableField(name="content", type=SearchFieldDataType.String)
        ]

        cors_options = CorsOptions(allowed_origins=["*"], max_age_in_seconds=60)

        index = SearchIndex(
            name=index_name,
            fields=fields,
            cors_options=cors_options
        )

        with SearchIndexClient(service_endpoint, AzureKeyCredential(key)) as search_index_client:
            search_index_client.create_index(index)
            logger.info(f"Index `{index_name}` created successfully.")
            pprint(f"[green]Index `{index_name}` created successfully.[/green]")
    except Exception as e:
        logger.error(f"Failed to create index `{index_name}`: {e}")
        pprint(f"[red]Failed to create index `{index_name}`: {e}[/red]")

def _create_vector_index():
    try:
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True, sortable=True),
            SimpleField(name="filetype", type=SearchFieldDataType.String, filterable=True, sortable=True, facetable=True),
            SimpleField(name="chunk_id", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
            SearchableField(name="filename", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SearchableField(name="url", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SearchableField(name="timestamp", type=SearchFieldDataType.String, filtrable=True, sortable=True),
            SearchableField(name="content", type=SearchFieldDataType.String, filtrable=True, sortable=True),
            SearchField(
                name="contentVector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="my-vector-search-profile"
            ),
            SearchField(
                name="imageVector",
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
            text_weights=TextWeights(weights={"content": 1.0})
        )

        scoring_profiles.append(scoring_profile)
        cors_options = CorsOptions(allowed_origins=["*"], max_age_in_seconds=60)

        semantic_config = SemanticConfiguration(
            name="my-semantic-configuration",
            prioritized_fields=SemanticField(
                title_field=SemanticField(field_name="filename"),
                # keyword_fields=[SemanticField(field_name="content")],
                content_fields=[SemanticField(field_name="content")]
            )
        )

        semantic_search = SemanticSearch(configurations=[semantic_config])

        with SearchIndexClient(service_endpoint, AzureKeyCredential(key)) as search_index_client:
            index = SearchIndex(
                name=index_vector_name,
                fields=fields,
                cors_options=cors_options,
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
        with SearchIndexerClient(service_endpoint, AzureKeyCredential(key)) as indexer_client:
            result = indexer_client.delete_data_source_connection(datasource_name)
            logger.info(f"Datasource connection `{datasource_name}` deleted successfully.")
            pprint(f"[green]Datasource connection `{datasource_name}` deleted successfully.[/green]")
    except Exception as e:
        logger.error(f"Error deleting datasource connection `{datasource_name}`: {e}")
        pprint(f"[red]Error deleting datasource connection `{datasource_name}`: {e}[/red]")

def _create_datasource_connection():
    try:
        container = SearchIndexerDataContainer(name=container_name)
        datasource_connection = SearchIndexerDataSourceConnection(
            name=datasource_name,
            type="azureblob",
            conntection_string=AZURE_STORAGE_CONNECTION_STRING,
            container=container
        )

        with SearchIndexerClient(service_endpoint, AzureKeyCredential(key)) as indexer_client:
            result = indexer_client.create_data_source_connection(datasource_connection)
            logger.info(f"Datasource connection `{datasource_name}` created successfully.")
            pprint(f"[green]Datasource Connection `{datasource_name}` created successfully.[/green]")

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
        logger.error(f"Error deleting skillset `{skillset_name}`: {e}")
        pprint(f"[red]Error deleting skillset `{skillset_name}`: {e}[/red]")  

def _create_skillset():
    try:
        id_input = InputFieldMappingEntry(name="id", source="/document/id")
        document_content_input = InputFieldMappingEntry(name="text", source="/document/content")
        url_input = InputFieldMappingEntry(name="url", source="/document/metadata_storage_path")
        filename_input = InputFieldMappingEntry(name="filename", source="/document/metadata_storage_name")
        filetype_input = InputFieldMappingEntry(name="filetype", source="/document/metadata_storage_content_type")
        images_input = InputFieldMappingEntry(name="images", source="/document/normalized_images/*")
        image_markers_input = InputFieldMappingEntry(name="image_markers", source="/document/normalized_images/*/width")
        image_input = InputFieldMappingEntry(name="images", source="/document/normalized_images/*")
        text_input = InputFieldMappingEntry(name="text", source="/document/text")

        text_output = OutputFieldMappingEntry(name="text", target_name="text")
        image_count_output = OutputFieldMappingEntry(name="image_count", target_name="image_count")

        document_cracking_skill = WebApiSkill(
            name="document-cracking-skill",
            description="Extract and clean text from documents.",
            inputs=[id_input, document_content_input, url_input, filename_input, filetype_input, image_markers_input],
            outputs=[text_output],
            context="/document",
            uri=AZURE_FUNCTION_APP_FOR_DOCUMENT_CRACKING_URL,
            timeout="PT230S",
            # batch_size=1,
            # degree_of_parallelism=1
        )

        extract_document_images_skill = WebApiSkill(
            name="extract-document-images-skill",
            description="Extract images from documents.",
            inputs=[id_input, text_input, url_input, filename_input, filetype_input, image_markers_input, images_input],
            outputs=[image_count_output],
            context="/document",
            uri=AZURE_FUNCTION_APP_FOR_EXTRACT_DOCUMENT_IMAGES_URL,
            timeout="PT230S",
            # batch_size=1,
            # degree_of_parallelism=1
        )

        skillset = SearchIndexerSkillset(
            name=skillset_name,
            skills=[document_cracking_skill, extract_document_images_skill],
            description="Skillset for extracting and processing document content and images."
        )

        with SearchIndexerClient(service_endpoint, AzureKeyCredential(key)) as search_indexer_client:
            search_indexer_client.create_skillset(skillset)
            logger.info(f"Skillset `{skillset_name}` created successfully.")
            pprint(f"[green]Skillset `{skillset_name}` created successfully.[/green]")

    except Exception as e:
        logger.error(f"Error creating skillset `{skillset_name}`: {e}")
        pprint(f"[red]Error creating skillset `{skillset_name}`: {e}[/red]")

def _create_indexer():
    try:
        configuration = IndexingParametersConfiguration(
            parsing_mode="default",
            data_to_extract="contentAndMetadata",
            image_action="generatedNormalizedImages"
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
            output_field_mappings=[
                FieldMapping(source_field_name="/document/text", target_field_name="content"),
                FieldMapping(source_field_name="/document/images_count", target_field_name="image_count")
            ]
        )

        with SearchIndexerClient(service_endpoint, AzureKeyCredential(key)) as search_indexer_client:
            search_indexer_client.create_indexer(indexer)
            logger.info(f"Indexer {indexer_name}` created successfully.")
            pprint(f"[green]Indexer `{indexer_name}` created successfully.[/green]")

    except Exception as e:
        logger.error(f"`{indexer_name}` creation failed: {e}")
        pprint(f"[red]`{indexer_name}` creation failed: {e}[/red]")








    


