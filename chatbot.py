import asyncio
import os
import nest_asyncio
from lightrag.llm.llama_index_impl import llama_index_complete
from llama_index.llms.groq import Groq
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger
from lightrag import LightRAG, QueryParam

# Enable debug logging
setup_logger("lightrag", level="DEBUG")
nest_asyncio.apply()

os.environ["NEO4J_URI"] = "neo4j+s://bc33095c.databases.neo4j.io"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "BpWzLNXymZI2invXPHkzOH-2DM_Vv5aQxdHXcCP2nAs"
os.environ["NEO4J_DATABASE"] = "neo4j"

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
        working_dir="graph_data",
        llm_model_func=llm_wrapper,
        graph_storage="Neo4JStorage",
        vector_storage=None,  # Disable vector storage
        addon_params={
            "use_vector_storage": False  # Explicitly disable vector storage usage
        }
    )

    await rag.initialize_storages()
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
            # Only use global and local modes since they rely on graph knowledge
            for mode in ["local", "global"]:
                try:
                    result = await rag.aquery(query, param=QueryParam(
                        mode=mode,
                        use_vector_search=False  # Explicitly disable vector search in queries
                    ))
                    print(f"Mode {mode}: {result}")
                except Exception as e:
                    print(f"Error in {mode} mode: {str(e)}")

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
