import os
from lightrag import LightRAG, QueryParam
from lightrag.llm.llama_index_impl import llama_index_complete
from llama_index.llms.groq import Groq

WORKING_DIR = "./data"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Create LLM instance
llm = Groq(model="llama3-70b-8192", api_key="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")

# Set Neo4j connection parameters
os.environ["NEO4J_URI"] = "neo4j+s://bc33095c.databases.neo4j.io"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "BpWzLNXymZI2invXPHkzOH-2DM_Vv5aQxdHXcCP2nAs"
os.environ["NEO4J_DATABASE"] = "neo4j"  # Explicitly set database name for Neo4j Aura

print("Neo4j Connection Parameters:")
print(f"URI: {os.getenv('NEO4J_URI')}")
print(f"Username: {os.getenv('NEO4J_USERNAME')}")
print(f"Password: {os.getenv('NEO4J_PASSWORD')}")
print(f"Database: {os.getenv('NEO4J_DATABASE')}")

# Create embedding function wrapper with proper attributes
class EmbeddingFunction:
    def __init__(self, llm_instance):
        self.llm = llm_instance
        self.embedding_dim = 384  # Set embedding dimension for LightRAG
        
    def __call__(self, text, **kwargs):
        try:
            return llama_index_complete(
                text,
                llm_instance=self.llm,
                **kwargs
            )
        except Exception as e:
            print(f"LLM Error: {str(e)}")
            raise

# Create the embedding function instance
embedding_func = EmbeddingFunction(llm)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=embedding_func,  # Use the embedding function instance
    graph_storage="Neo4JStorage",
    log_level="INFO",
)

with open("./facts.txt") as f:
    rag.insert(f.read())

# Perform naive search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))
)

# Perform local search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="local"))
)

# Perform global search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="global"))
)

# Perform hybrid search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid"))
)