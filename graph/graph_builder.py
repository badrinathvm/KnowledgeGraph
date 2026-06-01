from langgraph.graph import StateGraph, START, END
from nodes.nodes import retrieve, rerank, generate
from state.movie_state import MovieState

class GraphBuilder:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.workflow = StateGraph(MovieState)

    def build_graph(self):
        self.workflow.add_node("retrieve", retrieve)
        self.workflow.add_node("rerank", rerank)
        self.workflow.add_node("generate", generate)
        self.workflow.add_edge(START, "retrieve")
        # rerank node is only visited when strategy=="rerank"
        self.workflow.add_conditional_edges(
            "retrieve",
            lambda state: "rerank" if state.get("strategy") == "rerank" else "generate",
            {"rerank": "rerank", "generate": "generate"},
        )
        self.workflow.add_edge("rerank", "generate")
        self.workflow.add_edge("generate", END)
        return self

    def run(self, query: str, strategy: str = "default") -> dict:
        return self.build_graph().workflow.compile().invoke({
            "query": query,
            "strategy": strategy,
        })