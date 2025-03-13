import asyncio
import os
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from langchain_huggingface import HuggingFaceEmbeddings
from lightrag.kg.shared_storage import initialize_pipeline_status
from langchain_neo4j import Neo4jVector

# Load environment variables
load_dotenv()

# Get Neo4j credentials from environment variables
NEO4J_URL = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Define embedding function
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def embedding_function(text):
    return embeddings.embed_query(text)

def language_model_function(prompt):
    # Implement your text generation logic here
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2')

    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

async def initialize_rag():
    rag = LightRAG(
        working_dir="your/working/directory",
        embedding_func=embedding_function,
        llm_model_func=language_model_function,
        graph_storage="Neo4JStorage",
    )
    await rag.initialize_storages()
    return rag

async def main():
    rag = await initialize_rag()
    
    # Query the existing documents in Neo4j
    query_param = QueryParam(query="Your query here")
    response = rag.query(query_param)

    print("Query Response:", response)

if __name__ == "__main__":
    asyncio.run(main())
