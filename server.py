import asyncio
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models.models import GraphNode, GraphEdge, GraphResponse, QueryRequest, QueryResponse
from database.neo4j_database import neo4j_graph, plot_vector
from graph.graph_builder import GraphBuilder
from llm.llm import OpenAILLM

load_dotenv()

DEBUG = os.getenv("DEBUG", "true").lower() in ("1", "true", "yes")

app = FastAPI(title="Neo4j Graph API", debug=DEBUG)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Endpoints ----------

@app.get("/graph", response_model=GraphResponse)
async def get_graph():
    rows = await asyncio.to_thread(
        neo4j_graph.query,
        """
        MATCH (n) WHERE n:Movie OR n:Person OR n:Genre
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN
            elementId(n)   AS nodeId,
            labels(n)[0]   AS label,
            n              AS props,
            elementId(r)   AS relId,
            type(r)        AS relType,
            elementId(m)   AS targetId
        """
    )

    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []

    for row in rows:
        nid = row["nodeId"]
        if nid not in nodes:
            props = dict(row["props"])
            props.pop("plotEmbedding", None)
            for k, v in list(props.items()):
                if not isinstance(v, (str, int, float, bool, type(None))):
                    props[k] = str(v)

            label = row["label"]
            name = props.get("title") or props.get("name") or nid
            nodes[nid] = GraphNode(id=nid, label=label, name=name, properties=props)

        if row["relId"]:
            edges.append(GraphEdge(
                id=row["relId"],
                source=nid,
                target=row["targetId"],
                type=row["relType"],
            ))

    return GraphResponse(nodes=list(nodes.values()), edges=edges)


@app.post("/query", response_model=QueryResponse)
async def query_graph(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    # 1. Semantic search to find matching movie titles
    docs = await asyncio.to_thread(plot_vector.similarity_search, req.question, 6)
    titles = [doc.metadata.get("title") for doc in docs if doc.metadata.get("title")]

    # 2. Resolve titles → Neo4j elementIds
    highlighted_ids: list[str] = []
    for title in titles:
        rows = await asyncio.to_thread(
            neo4j_graph.query,
            "MATCH (m:Movie {title: $title}) RETURN elementId(m) AS id",
            params={"title": title},
        )
        highlighted_ids.extend(r["id"] for r in rows)

    # 3. Run LangGraph retrieve → generate for the LLM answer
    llm = OpenAILLM().get_llm()
    response = await asyncio.to_thread(GraphBuilder(llm).run, req.question)

    return QueryResponse(
        answer=response["result"],
        highlighted_node_ids=highlighted_ids,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=DEBUG,
        log_level="debug" if DEBUG else "info",
    )
