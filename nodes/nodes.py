from state.movie_state import MovieState
from database.neo4j_database import neo4j_graph, plot_vector
from llm.llm import OpenAILLM
from prompts.prompt import template


def retrieve(state: MovieState) -> dict:
    context = plot_vector.similarity_search(query=state["query"], k=6)
    return {"context": context}

def generate(state: MovieState) -> dict:
    query = state["query"]
    context = state["context"]
    llm = OpenAILLM().get_llm()
    chain = template | llm
    result = chain.invoke({"query": query, "context": context})
    return {"result": result.content}
