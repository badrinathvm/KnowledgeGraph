from langchain_core.prompts import ChatPromptTemplate


template = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful movie assistant. "
        "Answer questions about movies, including themes, plots, actors, directors, and genres. "
        "Use the provided context to answer. If the question is completely unrelated to movies, politely decline."
    )),
    ("human", "Context:\n{context}\n\nQuery: {query}"),
])


retrieval_query = """
    RETURN
        "Title: " + node.title + ", Plot: " + node.plot AS text,
        score,
        {
            title: node.title,
            imdbRating: node.imdbRating,
            released: toString(node.released),
            genres:    [ (node)-[:IN_GENRE]->(g)      | g.name ],
            actors:    [ (p)-[:ACTED_IN]->(node)       | p.name ],
            directors: [ (p)-[:DIRECTED]->(node)       | p.name ]
        } AS metadata
"""