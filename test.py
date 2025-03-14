import os
from dotenv import load_dotenv
import asyncio
import nest_asyncio
from lightrag import LightRAG
from lightrag.llm.llama_index_impl import llama_index_complete, llama_index_embed
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger, EmbeddingFunc
from lightrag import LightRAG, QueryParam

setup_logger("lightrag", level="INFO")

os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")


async def initialize_rag():
    
    llm = Groq(model="llama3-70b-8192", api_key="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")
    
    async def llm_wrapper(query, **kwargs):
        try:
            return await llama_index_complete(
                query,
                llm_instance=llm,
                **kwargs
            )
        except Exception as e:
            print(f"LLM Error: {str(e)}")
            raise
    rag = LightRAG(
        working_dir="data",
        llm_model_func=llm_wrapper,
        graph_storage="Neo4JStorage",
    )

    # Initialize database connections
    await rag.initialize_storages()
    # Initialize pipeline status for document processing
    await initialize_pipeline_status()

    return rag

async def main():
    try:
        # Read content from facts.txt
        with open("facts.txt", "r") as f:
            content = f.read()
        
        print(f"Loading content ({len(content)} characters)...")
        
        rag = await initialize_rag()
        rag.insert(content)
        print("Content inserted successfully")

        # Test queries
        test_queries = [
            "What is the recommended savings rate?"
        ]

        for query in test_queries:
            print(f"\nQuery: {query}")
            for mode in ["naive", "local", "global", "hybrid"]:
                try:
                    result = await rag.aquery(query, param=QueryParam(mode=mode))
                    print(f"Mode {mode}: {result}")
                except Exception as e:
                    print(f"Error in {mode} mode: {str(e)}")

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
