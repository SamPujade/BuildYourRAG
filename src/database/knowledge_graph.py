import sys
import yaml
import networkx as nx
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from typing import Dict, Any

sys.path.append("./src/")

from rag.embedding_model import Multilingual

with open("src/configs/config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)


class KnowledgeGraphRAG:
    def __init__(self, collection_name):
        # Initialize embedding model
        self.embedding_model = Multilingual()

        # Initialize graph
        self.graph = nx.DiGraph()

        self.chroma_client = chromadb.PersistentClient(
            path=config["dataset"]["CHROMA_DATA_PATH"],
            settings=Settings(),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )

    def add_node(self, node_id: str, content: str, metadata: Dict[str, Any] = None):
        """
        Add a node to the knowledge graph and embed its content

        Args:
            node_id (str): Unique identifier for the node
            content (str): Text content of the node
            metadata (dict, optional): Additional metadata for the node
        """
        # Add to networkx graph
        self.graph.add_node(node_id, content=content, metadata=metadata or {})

        # Generate embedding
        embedding = self.embedding_model.embed(content)

        # Ensure metadata is a non-empty dictionary
        metadata = metadata or {"file": "test"}

        # Add to ChromaDB
        self.collection.add(
            ids=[node_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata],  # Ensure that the metadata is a valid dictionary
        )

    def add_edge(self, source: str, target: str, relationship: str = None):
        """
        Add a directed edge between two nodes

        Args:
            source (str): Source node ID
            target (str): Target node ID
            relationship (str, optional): Type of relationship
        """
        self.graph.add_edge(source, target, relationship=relationship)

    def retrieve_similar_nodes(self, query: str, top_k: int = 5):
        """
        Retrieve most similar nodes to a given query.

        Args:
            query (str): Search query
            top_k (int): Number of top similar nodes to retrieve.

        Returns:
            List of most similar nodes.
        """
        # Generate query embedding
        query_embedding = self.embedding_model.embed(query)

        # Get the total number of nodes in the collection
        total_nodes = self.collection.count()

        # Adjust top_k if it exceeds the number of available nodes
        top_k = min(top_k, total_nodes)

        # Retrieve from ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )

        # Return the documents (already adjusted for n_results)
        return results.get("documents", [])
