import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import OpenAIEmbeddings


class Neo4jDatabase:
    def __init__(self):
        load_dotenv()
        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE")

    def get_graph(self) -> Neo4jGraph:
        try:
            return Neo4jGraph(
                url=self.uri,
                username=self.username,
                password=self.password,
                database=self.database,
            )
        except Exception as e:
            raise ValueError(f"Error connecting to Neo4j graph: {e}")

    def get_vector_index(self, graph: Neo4jGraph) -> Neo4jVector:
        try:
            embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
            return Neo4jVector.from_existing_index(
                embedding=embedding_model,
                graph=graph,
                index_name="moviePlots",
                embedding_node_property="plotEmbedding",
                text_node_property="plot",
            )
        except Exception as e:
            raise ValueError(f"Error connecting to Neo4j vector index: {e}")


_db = Neo4jDatabase()
neo4j_graph = _db.get_graph()
plot_vector = _db.get_vector_index(neo4j_graph)
