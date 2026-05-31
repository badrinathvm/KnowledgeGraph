from langgraph.graph import StateGraph, START, END
from nodes.nodes import retrieve, generate
from state.movie_state import MovieState

class GraphBuilder:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.workflow = StateGraph(MovieState)
        
    def build_graph(self):
        self.workflow.add_node("retrieve", retrieve)
        self.workflow.add_node("generate", generate)
        self.workflow.add_edge(START, "retrieve")
        self.workflow.add_edge("retrieve", "generate")
        self.workflow.add_edge("generate", END)
        return self

    def run(self, query: str) -> dict:
        return self.build_graph().workflow.compile().invoke({"query": query})