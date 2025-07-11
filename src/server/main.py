import os
import sys
import subprocess
from typing import List, Dict
from pydantic import BaseModel

from fastapi import FastAPI, File, UploadFile, Form
from dotenv import load_dotenv
import yaml
import chromadb

sys.path.append("./src/")

# local module imports
from models.generation import get_model_by_name, get_model_names
from models.embedding import get_model
from agents.agent import Agent


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = FastAPI()

with open("src/configs/config.yaml", "r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

CHROMA_DATA_PATH = config["dataset"]["CHROMA_DATA_PATH"]
UPLOAD_FOLDER = os.path.join(CHROMA_DATA_PATH, "upload/")
MODEL_FOLDER = config["generation"]["MODEL_FOLDER"]
DEFAULT_MODEL = config["generation"]["LLM"]
COLL_NAME = config["dataset"]["COLLECTION_NAME"]
EMB_MODEL_NAME = config["processing"]["EMBEDDING_MODEL"]
SIMILARITY = config["retrieval"]["SIMILARITY"]
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
embedding_function = get_model(EMB_MODEL_NAME).embedding_function
agent = Agent(config)


def give_permissions(folder):
    """
    Gives read and write permissions to all users for the specified folder.

    This function attempts to change the permissions of a given folder
    recursively to '777', granting read, write, and execute permissions
    to all users. It includes error handling for `subprocess.CalledProcessError`
    and other general exceptions.

    Args:
        folder (str): The path to the folder for which to change permissions.
    """
    try:
        command = ["chmod", "-R", "777", folder]
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to change permissions: {e}")
    except Exception as e:
        print(f"An error occured: {e}")


@app.post("/get-config/")
def get_config():
    """
    Retrieves the current configuration settings of the application.

    Returns:
        dict: A dictionary containing the current configuration.
    """
    return {
        "config": config,
    }


@app.post("/initial-message/")
def initial_message():
    """
    Retrieves the initial message set up for the agent.

    Returns:
        dict: A dictionary containing the agent's initial message.
    """
    return {
        "initial_message": agent.initial_message,
    }


class GenerationInput(BaseModel):
    """
    Represents the input structure for the text generation endpoint.

    Attributes:
        model_name (str): The name of the language model to use for generation.
        collection_name (str): The name of the document collection to use for
                               context.
        prompt_user (str): The user's input prompt.
        history (List[Dict[str, str]]): A list of dictionaries representing the
                                       conversation history.
        user_context (List[str]): A list of strings providing additional user
                                  context.
    """
    model_name: str
    collection_name: str
    prompt_user: str
    history: List[Dict[str, str]]
    user_context: List[str]


@app.post("/generate-response/")
def generate_response(body: GenerationInput):
    """
    Generates a response using a specified language model and context.

    Args:
        body (GenerationInput): The request body containing generation
                                parameters including model name, collection name,
                                user prompt, conversation history, and user
                                context.

    Returns:
        dict: A dictionary containing the generated output, the retrieved
              context, and the updated user context.
    """
    model = get_model_by_name(name=body.model_name, api_key=GOOGLE_API_KEY)
    agent.update_collection(body.collection_name, config)
    output, context, user_context = agent.predict(
        model, body.prompt_user, body.history, body.user_context
    )

    return {"output": output, "context": context, "user_context": user_context}


@app.post("/upload/")
async def upload_files(
    files: List[UploadFile] = File(...), collection_name: str = Form(...)
):
    """
    Uploads document files (e.g., PDF files) to a specified collection
    and processes them for indexing.

    Args:
        files (List[UploadFile]): A list of uploaded files.
        collection_name (str): The name of the collection where files will be
                               stored and indexed.

    Raises:
        Exception: If an error occurs during file processing.
    """
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": SIMILARITY},
    )

    for file in files:
        if file.filename != "":
            pdf_path = os.path.join(UPLOAD_FOLDER, collection_name, file.filename)
            content = await file.read()
            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
            with open(pdf_path, mode="wb") as w:
                w.write(content)

            try:
                agent.processing_function(collection, pdf_path, config)
            except Exception as e:
                os.remove(pdf_path)
                raise Exception(f"An error occurred during processing: {e}") from e


class DeleteInput(BaseModel):
    """
    Represents the input structure for deleting files from a collection.

    Attributes:
        files (List[str]): A list of filenames to be deleted.
        collection_name (str): The name of the collection from which to
                               delete files.
    """
    files: List[str]
    collection_name: str


@app.post("/delete-files")
def delete_files(body: DeleteInput):
    """
    Deletes specified files from a collection and removes their
    corresponding entries from the ChromaDB collection.

    Args:
        body (DeleteInput): The request body containing the list of files to
                            delete and the collection name.
    """
    collection = client.get_or_create_collection(
        name=body.collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": SIMILARITY},
    )

    for file in body.files:
        file_path = os.path.join(UPLOAD_FOLDER, body.collection_name, file)
        if os.path.exists(file_path):
            os.remove(file_path)

    elements = collection.get()
    ids_to_remove = [
        id
        for id, metadata in zip(elements["ids"], elements["metadatas"])
        if metadata.get("from") in body.files
    ]
    if ids_to_remove:
        collection.delete(ids=ids_to_remove)


@app.post("/delete-collection/")
def delete_collection(body: DeleteInput):
    """
    Deletes an entire collection, including all files within its associated
    folder and the collection itself from the ChromaDB database.

    Args:
        body (DeleteInput): The request body containing the name of the
                            collection to delete.
    """
    collection = client.get_or_create_collection(
        name=body.collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": SIMILARITY},
    )

    folder_path = os.path.join(UPLOAD_FOLDER, body.collection_name)
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            os.remove(os.path.join(folder_path, file))
        os.rmdir(folder_path)

    ids = collection.get()["ids"]
    if ids:
        collection.delete(ids=ids)
    client.delete_collection(body.collection_name)


class CollectionInput(BaseModel):
    """
    Represents the input structure for operations involving a collection name.

    Attributes:
        collection_name (str): The name of the collection.
    """
    collection_name: str


@app.post("/create-collection/")
def create_collection(body: CollectionInput):
    """
    Creates a new collection and its corresponding upload folder.

    If the folder for the new collection does not exist, it will be created
    and appropriate permissions will be set. The collection will also be
    initialized in ChromaDB.

    Args:
        body (CollectionInput): The request body containing the name of the
                                collection to create.
    """
    folder_path = os.path.join(UPLOAD_FOLDER, body.collection_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        give_permissions(folder_path)

    collection = client.create_collection(
        name=body.collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": SIMILARITY},
    )


@app.post("/get-names/")
def get_names():
    """
    Retrieves available model names and collection names, along with
    their default selections.

    Returns:
        dict: A dictionary containing lists of model names and collection
              names, and their respective default indices.
    """
    model_names = get_model_names()
    model_index = (
        model_names.index(DEFAULT_MODEL) if DEFAULT_MODEL in model_names else 0
    )

    coll_list = [
        name for name in os.listdir(UPLOAD_FOLDER)
        if os.path.isdir(os.path.join(UPLOAD_FOLDER, name))
    ]
    coll_index = coll_list.index(COLL_NAME) if COLL_NAME in coll_list else 0

    return {
        "model_names": model_names,
        "model_index": model_index,
        "collection_list": coll_list,
        "collection_index": coll_index,
    }


@app.post("/list-files/")
def list_files(body: CollectionInput):
    """
    Lists all files within a specified collection's upload folder.

    Args:
        body (CollectionInput): The request body containing the name of the
                                collection.

    Returns:
        dict: A dictionary containing a list of filenames within the
              specified folder.
    """
    folder_path = os.path.join(UPLOAD_FOLDER, body.collection_name)
    files_list = []
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        files_list = os.listdir(folder_path)

    return {
        "files_list": files_list,
    }


class ConfigInput(BaseModel):
    """
    Represents the input structure for updating configuration settings.

    Attributes:
        temperature (float): The temperature setting for text generation.
        layout (bool): Flag indicating whether layout processing is enabled.
        sub_chunking (bool): Flag indicating whether sub-chunking is enabled.
        figure_extraction (bool): Flag indicating whether figure extraction
                                  is enabled.
        top_k (int): The number of top relevant documents to retrieve.
    """
    temperature: float
    layout: bool
    sub_chunking: bool
    figure_extraction: bool
    top_k: int


@app.post("/update-config/")
def update_config(body: ConfigInput):
    """
    Updates various configuration settings of the application.

    Args:
        body (ConfigInput): The request body containing the new configuration
                            values for temperature, layout, sub-chunking,
                            figure extraction, and top_k.
    """
    config["generation"]["TEMPERATURE"] = body.temperature
    config["processing"]["LAYOUT"] = body.layout
    config["processing"]["SUB_CHUNKING"] = body.sub_chunking
    config["processing"]["FIGURE_EXTRACTION"] = body.figure_extraction
    config["retrieval"]["TOP_K"] = body.top_k