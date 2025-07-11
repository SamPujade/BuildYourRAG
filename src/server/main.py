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
    Gives read and write permissions all users
    to folders for the newly created collections
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
    return {
        "config": config,
    }


@app.post("/initial-message/")
def initial_message():
    return {
        "initial_message": agent.initial_message,
    }


class GenerationInput(BaseModel):
    model_name: str
    collection_name: str
    prompt_user: str
    history: List[Dict[str, str]]
    user_context: List[str]


@app.post("/generate-response/")
def generate_response(body: GenerationInput):
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
    Upload documents from uploaded PDF files.

    Args:
        uploaded_files (list): List of uploaded PDF files.
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
            with open(pdf_path, mode="wb") as w:
                w.write(content)

            try:
                agent.processing_function(collection, pdf_path, config)
            except Exception as e:
                os.remove(pdf_path)
                raise Exception(f"An error occurred during processing: {e}") from e


class DeleteInput(BaseModel):
    files: List[str]
    collection_name: str


@app.post("/delete-files")
def delete_files(body: DeleteInput):
    collection = client.get_or_create_collection(
        name=body.collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": SIMILARITY},
    )

    for file in body.files:
        file_path = os.path.join(UPLOAD_FOLDER, body.collection_name, file)
        os.remove(file_path)

    elements = collection.get()
    ids_to_remove = [
        id
        for id, metadata in zip(elements["ids"], elements["metadatas"])
        if metadata["from"] in body.files
    ]
    if ids_to_remove:
        collection.delete(ids=ids_to_remove)


@app.post("/delete-collection/")
def delete_collection(body: DeleteInput):
    """
    Delete a collectiom : remove each file in the folder, remove the folder, and delete the collection from the chroma db database.

    Args:
        folder (str): Selected upload folder.
    """
    collection = client.get_or_create_collection(
        name=body.collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": SIMILARITY},
    )

    folder_path = os.path.join(UPLOAD_FOLDER, body.collection_name)
    for file in os.listdir(folder_path):
        os.remove(os.path.join(folder_path, file))
    os.rmdir(folder_path)

    ids = collection.get()["ids"]
    if ids:
        collection.delete(ids)
    client.delete_collection(body.collection_name)


class CollectionInput(BaseModel):
    collection_name: str


@app.post("/create-collection/")
def create_collection(body: CollectionInput):
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
    model_names = get_model_names()
    model_index = (
        model_names.index(DEFAULT_MODEL) if DEFAULT_MODEL in model_names else 0
    )

    coll_list = os.listdir(UPLOAD_FOLDER)
    coll_index = coll_list.index(COLL_NAME) if COLL_NAME in coll_list else 0

    return {
        "model_names": model_names,
        "model_index": model_index,
        "collection_list": coll_list,
        "collection_index": coll_index,
    }


@app.post("/list-files/")
def list_files(body: CollectionInput):
    folder_path = os.path.join(UPLOAD_FOLDER, body.collection_name)
    files_list = os.listdir(folder_path)

    return {
        "files_list": files_list,
    }


class ConfigInput(BaseModel):
    temperature: float
    layout: bool
    sub_chunking: bool
    figure_extraction: bool
    top_k: int


@app.post("/update-config/")
def update_config(body: ConfigInput):
    config["generation"]["TEMPERATURE"] = body.temperature
    config["processing"]["LAYOUT"] = body.layout
    config["processing"]["SUB_CHUNKING"] = body.sub_chunking
    config["processing"]["FIGURE_EXTRACTION"] = body.figure_extraction
    config["retrieval"]["TOP_K"] = body.top_k
