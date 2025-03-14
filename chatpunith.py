import asyncio
import nest_asyncio
from lightrag import LightRAG
from lightrag.llm.llama_index_impl import llama_index_complete, llama_index_embed
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger, EmbeddingFunc
from lightrag import LightRAG, QueryParam

# Enable debug logging
setup_logger("lightrag", level="DEBUG")
nest_asyncio.apply()

async def initialize_rag():
    llm = Groq(model="llama3-70b-8192", api_key="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")
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

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
        working_dir="vectorized_data",
        llm_model_func=llm_wrapper,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=8192,
            func=lambda texts: llama_index_embed(texts, embed_model=embed_model)
        ),
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
