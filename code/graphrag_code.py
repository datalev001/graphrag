### Microsoft GraphRAG: Batch Search (Command Line Interface)

pip install graphrag
conda create -n graphrag-env python=3.11
conda activate graphrag-en
python -m graphrag.index --init --root C:\backupcgi\ragtest
python -m graphrag.index --root C:\backupcgi\ragtest

# Try the following GRAPHRAG commands

python -m graphrag.query --root  C:\backupcgi\ragtest --method global "What are the top themes in this story?"

python -m graphrag.query --root  C:\backupcgi\ragtest --method local "What are the top themes in this story?"

python -m graphrag.query --root  C:\backupcgi\ragtest --method global "using less than 100 words to describe the top themes in this story"

python -m graphrag.query --root  C:\backupcgi\ragtest --method local "using less than 100 words to describe the top themes in this story"


######Structured Microsoft GraphRAG Local query###############
llm = ChatOpenAI(
    api_key="####",
    model="gpt-4",
    deployment_name = "gpt4",
    api_type= OpenaiApiType.AzureOpenAI,
    api_base="https://####",
    max_retries=20,
    api_version = "2023-12-01-preview"
)

token_encoder = tiktoken.get_encoding("cl100k_base")

text_embedder = OpenAIEmbedding(
    api_key="###",
    api_base="https://###",
    api_type=OpenaiApiType.AzureOpenAI,
    model="text-embedding-3-large",
    deployment_name="textembedding3large",
    api_version = "2023-12-01-preview",
    max_retries=20,
)

INPUT_DIR = "../ragtest/output/20240811-142734/artifacts"
LANCEDB_URI = f"{INPUT_DIR}/lancedb"
entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")

context_builder = LocalSearchMixedContext(
    community_reports=reports,
    text_units=text_units,
    entities=entities,
    relationships=relationships,
    covariates=None,
    entity_text_embeddings=description_embedding_store,
    embedding_vectorstore_key=EntityVectorStoreKey.ID,
    text_embedder=text_embedder,
    token_encoder=token_encoder,
)

local_context_params = {
    "text_unit_prop": 0.5,
    "community_prop": 0.1,
    "conversation_history_max_turns": 5,
    "conversation_history_user_turns_only": True,
    "top_k_mapped_entities": 10,
    "top_k_relationships": 10,
    "include_entity_rank": True,
    "include_relationship_weight": True,
    "include_community_rank": False,
    "return_candidate_context": False,
    "embedding_vectorstore_key": EntityVectorStoreKey.ID,
    "max_tokens": 12_000,
}


search_engine = LocalSearch(
    llm=llm,
    context_builder=context_builder,
    token_encoder=token_encoder,
    llm_params=llm_params,
    context_builder_params=local_context_params,
    response_type="multiple paragraphs",
)

result = await search_engine.asearch(
    "What are the major topics of this story?"
)

######Structured  Microsoft GraphRAG Global query###############
COMMUNITY_LEVEL = 2
entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")

reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

context_builder = GlobalCommunityContext(
    community_reports=reports,
    entities=entities,  
    token_encoder=token_encoder,
)

context_builder_params = {
    "use_community_summary": False,
    "shuffle_data": True,
    "include_community_rank": True,
    "min_community_rank": 0,
    "community_rank_name": "rank",
    "include_community_weight": True,
    "community_weight_name": "occurrence weight",
    "normalize_community_weight": True,
    "max_tokens": 12_000,
    "context_name": "Reports",
}

map_llm_params = {
    "max_tokens": 1000,
    "temperature": 0.7,
    "response_format": {"type": "json_object"},
}

reduce_llm_params = {
    "max_tokens": 2000,
    "temperature": 0.7,
}

search_engine = GlobalSearch(
    llm=llm,
    context_builder=context_builder,
    token_encoder=token_encoder,
    max_data_tokens=12_000,
    map_llm_params=map_llm_params,
    reduce_llm_params=reduce_llm_params,
    allow_general_knowledge=False,
    json_mode=True,
    context_builder_params=context_builder_params,
    concurrent_coroutines=32,
    response_type="multiple paragraphs",
)


########Neo4j + LangChain GraphRAG########
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from neo4j import GraphDatabase
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_community.graphs import Neo4jGraph

os.environ["NEO4J_URI"] = "bolt://localhost:7687"  # Default URI for Neo4j Desktop
os.environ["NEO4J_USERNAME"] = "neo4j"  # Replace with your Neo4j username
os.environ["NEO4J_PASSWORD"] = "12345678"  # Replace with your Neo4j password

# Connect to the Neo4j graph database
graph = Neo4jGraph(refresh_schema=False)

loader = TextLoader("../mechanics.txt", encoding='UTF-8')
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
    length_function=len,
)

documents = text_splitter.split_documents(docs)

llm = AzureChatOpenAI(
    openai_api_version="2023-12-01-preview",
    azure_deployment="gpt4",
)

llm_transformer = LLMGraphTransformer(
  llm=llm, 
)

# Form documents 
graph_doc = llm_transformer.convert_to_graph_documents(documents)

# Add into Neo4j
graph.add_graph_documents(graph_doc, baseEntityLabel=True, include_source=True)

from yfiles_jupyter_graphs import GraphWidget

def showGraph():
    driver = GraphDatabase.driver(uri = "bolt://localhost:7687",
         auth = ("neo4j","12345678"))
    session = driver.session()
    widget = GraphWidget(graph = session.run("MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t").graph())
    widget.node_label_mapping = 'id'
    return widget

showGraph()

embeddings = AzureOpenAIEmbeddings(
    deployment="textembedding3large",
    model="text-embedding-3-large",
    azure_endpoint="https://###",
    openai_api_type="azure",
)

vector_index = Neo4jVector.from_existing_graph(
    embeddings,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

vector_retriever = vector_index.as_retriever()

def graph_retriever(question: str):
    vector_data = [el.page_content for el in vector_retriever.invoke(question)]
    final_data = f"""
    vector data:
    {"#Document ".join(vector_data)}
    """
    return final_data

question = "How is Newton's Third Law related to Principles of Mechanics?

template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": graph_retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

chain.invoke(input= question)


######traditional RAG#########
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI

OPENAI_API_KEY = "###"
OPENAI_API_BASE = "https://###"
OPENAI_API_VERSION = "2023-12-01-preview"
OPENAI_DEPLOYMENT_NAME = "gpt4"
MODEL_NAME = "gpt-4"


#  Splits the loaded documents into smaller text chunks with specified size and overlap.
loader = TextLoader("../dinner.txt", encoding = 'UTF-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = text_splitter.split_documents(documents)


emb_model = AzureOpenAIEmbeddings(deployment='textembedding3large',
                              model='text-embedding-3-large',
                              openai_api_key=OPENAI_API_KEY,
                              azure_endpoint=OPENAI_API_BASE,
                              openai_api_type="azure")


#Retrieves the top-k relevant document chunks for a given question using max marginal relevance search from the vector #database and concatenates their content.
def get_chunks(question,k):
  loaded_vectordb = Chroma(persist_directory= "../chroma_db", embedding_function= emb_model)
  docs = loaded_vectordb.max_marginal_relevance_search(question, k=5)
  chunks = ' '.join([chunk.page_content for chunk in docs])
  return chunks

# Feeds data into a vector-based database Chroma using document embeddings generated with Azure OpenAI.
vectordb = Chroma.from_documents(
    documents=chunks, 
    embedding=AzureOpenAIEmbeddings(deployment='textembedding3large',
    model='text-embedding-3-large', 
    azure_endpoint=OPENAI_API_BASE,
    openai_api_key=OPENAI_API_KEY,
    openai_api_type="azure"),
    persist_directory= "../chroma_db")

question = "How did the events of the family dinner strengthen the relationships between the family members?"

OPENAI_DEPLOYMENT_NAME = "gpt4"
MODEL_NAME = "gpt-4"

model = AzureChatOpenAI(azure_endpoint=OPENAI_API_BASE, openai_api_version= OPENAI_API_VERSION,
azure_deployment= OPENAI_DEPLOYMENT_NAME, openai_api_key=OPENAI_API_KEY, openai_api_type= "azure")

retrieved = get_chunks(question,1)

template = """You are an assistant for question-answering tasks. """

hint = """ Use the following pieces of retrieved context to answer the question. 
  Use up to 120 words maximum and keep the answer concise.
  use the following the context: """ + retrieved  +  "Answer the question:"  + question 

prompt = template +  hint 
response = model.predict(prompt)

print (response)
