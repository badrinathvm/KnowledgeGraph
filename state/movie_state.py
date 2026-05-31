from typing_extensions import TypedDict, List

class MovieState(TypedDict):
    query: str
    context: List[dict]
    result: str