def group_sub_chunks(collection):
    """Groups sub-chunks in a ChromaDB collection into complete chunks.

    Args:
        collection: The ChromaDB collection object.

    Returns:
        tuple: A tuple of grouped chunk documents and their metadata.
    """
    elements = collection.get()
    chunks, indexes = [], []
    metadatas = []

    for i in range(len(elements["documents"])):
        # If sub chunk:
        if "chunk" in elements["metadatas"][i]:
            chunk_index = elements["metadatas"][i]["chunk"]
            if chunk_index not in indexes:
                subchunks = collection.get(where={"chunk": chunk_index})["documents"]
                chunks.append("".join(subchunks))
                indexes.append(chunk_index)
                metadatas.append(
                    {k: v for k, v in elements["metadatas"][i].items() if k != "chunk"}
                )

        # If not a not sub chunk:
        else:
            chunks.append(elements["documents"][i])
            metadatas.append(
                {k: v for k, v in elements["metadatas"][i].items() if k != "chunk"}
            )

    return chunks, metadatas


def delete_collection(client, collection_name):
    """Deletes a specified collection from the ChromaDB client.

    Args:
        client: The ChromaDB client object.
        collection_name (str): The name of the collection to be deleted.
    """
    collection = client.get_collection(collection_name)
    ids = collection.get()["ids"]
    if ids:
        collection.delete(ids)
    client.delete_collection(collection_name)
    print(f"Collection succesfully deleted : {collection_name}")