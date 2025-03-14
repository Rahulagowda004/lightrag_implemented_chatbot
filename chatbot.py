import asyncio
from lightrag import LightRAG
<<<<<<< HEAD
from lightrag.llm.llama_index_impl import llama_index_embed
from llama_index.embeddings.groq import GroqEmbedding  # Replace OpenAIEmbedding
from llama_index.llms.groq import Groq  # Replace OpenAI
=======
from lightrag.llm.llama_index_impl import llama_index_complete_if_cache, llama_index_embed
from llama_index.embeddings.
from llama_index.llms.groq import Groq
>>>>>>> 5ad9460b92ad24e108f8af079eb23fd4baaf4ff4
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

# Setup log handler for LightRAG
setup_logger("lightrag", level="INFO")

# Initialize Groq LLM and embedding
groq_model = Groq(model="llama3-8b-8192")  # Replace with your model
embed_model = GroqEmbedding(model="bge-small-en")  # Adjust embedding model if needed

async def initialize_rag():
    rag = LightRAG(
        working_dir="your/path",
        llm_model_func=lambda prompt: groq_model.complete(prompt),  # Use Groq's completion
        embedding_func=lambda texts: llama_index_embed(texts, embed_model=embed_model),  # Groq embedding
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())

    with open("./book.txt", "r", encoding="utf-8") as f:
        rag.insert(f.read())

    # Perform different search modes
    for mode in ["naive", "local", "global", "hybrid"]:
        print(rag.query("What are the top themes in this story?", param=QueryParam(mode=mode)))

if __name__ == "__main__":
    main()
