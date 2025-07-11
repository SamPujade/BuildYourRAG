import chromadb

from models.embedding import get_model


class Retriever:
    def __init__(self, collection_name, config):
        self.config = config
        self.data_path = self.config["dataset"]["CHROMA_DATA_PATH"]
        self.top_k = config["retrieval"]["TOP_K"]
        self.similarity = config["retrieval"]["SIMILARITY"]
        client = chromadb.PersistentClient(path=self.data_path)
        embedding_model = get_model(self.config["processing"]["EMBEDDING_MODEL"])

        self.collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_model.embedding_function,
            metadata={"hnsw:space": self.similarity},
        )
        if self.collection is None:
            raise ValueError("Collection not found")

    def retrieve(self, query_text):
        sub_result = self.collection.query(
            query_texts=[query_text],
            n_results=self.top_k * 3,
            # include=["documents", "distances", "metadatas"],
        )

        context, indexes = [], []
        i = 0

        while len(context) < self.top_k and i < len(sub_result["documents"][0]):
            metadatas = sub_result["metadatas"][0][i]

            # Add metadatas information
            metadata_information = ""
            for key in metadatas:
                if key not in ["from", "type", "chunk"]:
                    metadata_information += f"{key}: {metadatas[key]}\n"
            metadata_information += "\n" if metadata_information else ""

            # If sub chunk:
            if "chunk" in metadatas:
                chunk_index = metadatas["chunk"]
                if chunk_index not in indexes:
                    subchunks = self.collection.get(where={"chunk": chunk_index})[
                        "documents"
                    ]
                    context.append(metadata_information + "".join(subchunks))
                    indexes.append(chunk_index)

            # If not a not sub chunk:
            else:
                context.append(metadata_information + sub_result["documents"][0][i])

            i += 1

        return context
