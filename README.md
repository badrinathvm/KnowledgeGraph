# KnowledgeGraph

A FastAPI backend that combines a Neo4j graph database with LangGraph and OpenAI to answer natural language questions about movies. It performs semantic vector search over movie plots, resolves matching nodes in the graph, and runs a retrieve → generate pipeline to produce answers with highlighted graph nodes.

## Architecture

```
server.py               FastAPI app — /graph and /query endpoints
├── database/
│   └── neo4j_database.py   Neo4j connection + vector index (shared singletons)
├── graph/
│   └── graph_builder.py    LangGraph workflow (build + run)
├── nodes/
│   └── nodes.py            retrieve and generate node functions
├── llm/
│   └── llm.py              OpenAI LLM factory
├── models/
│   └── models.py           Pydantic request/response models
├── prompts/
│   └── prompt.py           ChatPromptTemplate for the generate node
└── state/
    └── movie_state.py      LangGraph state schema (query, context, result)
```

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- A running Neo4j instance with the movie dataset loaded and a `moviePlots` vector index
- An OpenAI API key

## Setup

1. **Clone the repo and install dependencies**

   ```bash
   uv sync
   ```

2. **Configure environment variables**

   Copy `.env.example` to `.env` and fill in your values:

   ```env
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password
   NEO4J_DATABASE=neo4j
   OPENAI_API_KEY=sk-...
   ```

3. **Run the server**

   ```bash
   uv run uvicorn server:app --reload
   ```

   Or via the debug configuration in Cursor / VS Code (`Debug API Server`).

## API

### `GET /graph`

Returns all `Movie`, `Person`, and `Genre` nodes and their relationships.

**Response**
```json
{
  "nodes": [{ "id": "...", "label": "Movie", "name": "The Matrix", "properties": {} }],
  "edges": [{ "id": "...", "source": "...", "target": "...", "type": "ACTED_IN" }]
}
```

### `POST /query`

Answers a natural language movie question and returns the IDs of relevant graph nodes.

**Request**
```json
{ "question": "Who directed Inception?" }
```

**Response**
```json
{
  "answer": "Inception was directed by Christopher Nolan.",
  "highlighted_node_ids": ["4:abc123...", "4:def456..."]
}
```

## How it works

1. **Semantic search** — the question is embedded and searched against the `moviePlots` Neo4j vector index to find the most relevant movies.
2. **Node resolution** — matched movie titles are resolved to their Neo4j element IDs for graph highlighting.
3. **LangGraph pipeline** — a `retrieve → generate` graph fetches context via vector similarity and feeds it to GPT-4o to produce the final answer.
