import asyncio
import os
import nest_asyncio
from lightrag.llm.llama_index_impl import llama_index_complete
from llama_index.llms.groq import Groq
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger
from lightrag import LightRAG, QueryParam
from sentence_transformers import SentenceTransformer

# Enable debug logging
setup_logger("lightrag", level="DEBUG")
nest_asyncio.apply()

os.environ["NEO4J_URI"] = "neo4j+s://bc33095c.databases.neo4j.io"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "BpWzLNXymZI2invXPHkzOH-2DM_Vv5aQxdHXcCP2nAs"
os.environ["NEO4J_DATABASE"] = "neo4j"

class EmbeddingWrapper:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384  # This model's output dimension

    async def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

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
        working_dir="vectorized_data",
        llm_model_func=llm_wrapper,
        graph_storage="Neo4JStorage",
        vector_storage="NanoVectorDBStorage",
        embedding_func=EmbeddingWrapper(),
        addon_params={
            "use_vector_storage": True,
            "store_response": True,
            "enable_knowledge_extraction": True,
            "extract_entities": True,
            "extract_relationships": True,
            "kg_storage_type": "Neo4JStorage"
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
        
        rag = await initialize_rag()
        rag.insert(content)
        print("Content inserted successfully")

        try:
            while True:
                query = input("you: ")
                if query == "bye":
                    break
                result = await rag.aquery(
                    query, 
                    param=QueryParam(
                        mode="global",
                        response_type="Bullet Points",
                    )
                )
                response_text = str(result) if result else "No response generated"
                print("*"*150)
                print("you:", query)
                print("bot:", response_text)
                print("*"*150)
                
                with open("output.txt", "a") as f:
                    f.write(f"\nQuery: {query}\nMode: global: {response_text}\n")
                    
                try:
                    rag.insert(response_text)        
                except Exception as e:
                    print(f"Failed to store via lightrag: {str(e)}")
                
        except Exception as e:
            print(f"Error in {"global"} mode: {str(e)}")

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())