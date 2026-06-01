from typing_extensions import TypedDict, List

class MovieState(TypedDict):
    query: str
    strategy: str   # "default" | "mmr" | "rerank"
    context: List[dict]
    result: str