from state.movie_state import MovieState
from database.neo4j_database import neo4j_graph, plot_vector
from llm.llm import OpenAILLM
from prompts.prompt import template

_cross_encoder = None

def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def retrieve(state: MovieState) -> dict:
    query = state["query"]
    strategy = state.get("strategy", "default")

    if strategy == "mmr":
        # Option A: fetch 20, return 6 diverse results (no near-duplicates)
        context = plot_vector.max_marginal_relevance_search(query=query, k=6, fetch_k=20)
    elif strategy == "rerank":
        # Option B: fetch 20 candidates; rerank node will trim to top 6
        context = plot_vector.similarity_search(query=query, k=20)
    else:
        # default: standard similarity search
        context = plot_vector.similarity_search(query=query, k=6)

    return {"context": context}


def rerank(state: MovieState) -> dict:
    # Option B: cross-encoder scores each candidate against the query, keeps top 6
    query = state["query"]
    candidates = state["context"]
    encoder = _get_cross_encoder()
    scores = encoder.predict([[query, doc.page_content] for doc in candidates])
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return {"context": [doc for _, doc in ranked[:6]]}


def generate(state: MovieState) -> dict:
    query = state["query"]
    context = state["context"]
    llm = OpenAILLM().get_llm()
    chain = template | llm
    result = chain.invoke({"query": query, "context": context})
    return {"result": result.content}
