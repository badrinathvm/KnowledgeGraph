from pydantic import BaseModel


class GraphNode(BaseModel):
    id: str
    label: str
    name: str
    properties: dict


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    type: str


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class QueryRequest(BaseModel):
    question: str
    strategy: str = "default"  # "default" | "mmr" | "rerank"


class QueryResponse(BaseModel):
    answer: str
    highlighted_node_ids: list[str]
