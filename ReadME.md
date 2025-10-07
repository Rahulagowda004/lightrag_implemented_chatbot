# LightRAG Implemented Chatbot

A Python-based chatbot leveraging Retrieval-Augmented Generation (RAG) with vector and knowledge graph storage, powered by Llama 3 LLM and Neo4j graph database. This chatbot can ingest a knowledge base and answer user queries with concise, context-aware responses.

## Features

- **Retrieval-Augmented Generation (RAG):** Uses LightRAG to combine vector and graph storage for efficient knowledge retrieval and reasoning.
- **Large Language Model Integration:** Utilizes Llama 3 (`llama3-70b-8192`) via Groq API for advanced natural language understanding and generation.
- **Knowledge Graph Storage:** Stores and manages knowledge using Neo4j, enabling entity and relationship extraction.
- **Vector Database:** Fast semantic search and embedding via NanoVectorDB and SentenceTransformers (`all-MiniLM-L6-v2`).
- **Continuous Learning:** Can store chat responses and expand its knowledge base dynamically.
- **Debug Logging:** Detailed debug logs for easier troubleshooting.

## Quickstart

### Prerequisites

- Python 3.8+
- Neo4j Aura database credentials (or local Neo4j instance)
- Groq API key for Llama 3 model
- Required Python packages (see below)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rahulagowda004/lightrag_implemented_chatbot.git
   cd lightrag_implemented_chatbot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   _If `requirements.txt` is missing, install manually:_
   ```bash
   pip install sentence-transformers neo4j nest_asyncio
   ```

3. **Configure Environment Variables:**
   Set these in your shell or `.env` file:
   ```
   NEO4J_URI=neo4j+s://<your-neo4j-uri>
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=<your-password>
   NEO4J_DATABASE=neo4j
   GROQ_API_KEY=<your-groq-api-key>
   ```

4. **Prepare your knowledge base:**
   - Add information to `facts.txt` (plain text, each fact on a new line).

### Usage

#### Interactive Chat

Run the chatbot in interactive mode:

```bash
python testing.py
```

- Enter your queries in the terminal.
- Type `bye` to exit.

#### Scripted Query

Run predefined queries from `chatbot.py`:

```bash
python chatbot.py
```

This will load the knowledge base and perform test queries.

### Example (from logs)

```
Query: who are you
Mode global: I am the AI Assistant, designed to provide information and assist with queries based on the provided Knowledge Base.
```

```
Query: my name is rahul keep it in mind
Mode: global: Hello Rahul! I'm happy to assist you with any questions you may have.
```

## Files

- `chatbot.py` — Core chatbot logic and main function.
- `testing.py` — Interactive chat handler.
- `facts.txt` — Editable knowledge base.
- `output.txt` — Stores chat logs and responses.
- `Neo4j/neo4j_impl.py` — Neo4j integration for graph storage.
- `readthis.txt` — Note about custom library modifications.

## Customization & Notes

- The LightRAG library is **locally modified**; future updates may require debugging (`readthis.txt`).
- You can expand the knowledge base by editing `facts.txt`.
- Review and adjust logging and storage parameters as needed.

## License

_No explicit license is provided. Please contact the repository owner for usage permissions._

## Author

- [Rahulagowda004](https://github.com/Rahulagowda004)

---

For more details, see the [repository](https://github.com/Rahulagowda004/lightrag_implemented_chatbot).
