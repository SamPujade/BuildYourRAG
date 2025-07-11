import os
import sys
from dotenv import load_dotenv

# third-party imports
from pdf2image import convert_from_path
from PyPDF2 import PdfReader


sys.path.append("./src/")

from models.generation import GeminiFlash

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def token_len(chunk):
    """Calculates the number of tokens (words) in a given chunk of text.

    Args:
        chunk (str): The text chunk.

    Returns:
        int: The number of tokens in the chunk.
    """
    return len(chunk.split())


def basic_chunking(texts, config):
    """Performs basic text chunking based on a separator and max chunk size.

    Args:
        texts (list): List of text strings to chunk.
        config (dict): Configuration dictionary with processing parameters.

    Returns:
        tuple: A tuple containing:
            - list: A list of text chunks (str).
            - list: A list of labels, all "text" for basic chunking.
    """
    separator = config["processing"]["SEPARATOR"]
    max_chunk_size = config["processing"]["MAX_CHUNK_SIZE"]
    chunks, current_chunk = [], ""

    for text in texts:
        entries = text.split(separator)
        entries = [entry.replace("-\n", "") for entry in entries]

        for entry in entries:
            if not current_chunk:
                current_chunk = entry
                continue
            if token_len(current_chunk + " " + entry) > max_chunk_size:
                chunks.append(current_chunk)
                current_chunk = entry
            else:
                current_chunk += "\n" + entry

    if current_chunk:
        chunks.append(current_chunk)

    return chunks, ["text" for _ in chunks]


def sub_chunking(chunks, labels):
    """Splits each chunk into smaller sub-chunks.

    Args:
        chunks (list of str): List of input chunks.
        labels (list): Corresponding labels for the input chunks.

    Returns:
        tuple: A tuple containing:
            - list: List of resulting sub-chunks.
            - list: Corresponding labels for each sub-chunk.
            - list: List of indexes indicating the parent chunk of each sub-chunk.
    """
    max_token_length = 64
    sub_chunks, new_labels, chunk_indexes = [], [], []

    for chunk_index, (chunk, label) in enumerate(zip(chunks, labels)):
        current_subchunk = ""

        for part in chunk.split("\n"):
            if token_len(current_subchunk + " " + part) > max_token_length:
                if current_subchunk:  # Only add if it's non-empty
                    sub_chunks.append(current_subchunk)
                    new_labels.append(label)
                    chunk_indexes.append(chunk_index)
                current_subchunk = part + "\n"
            else:
                current_subchunk += part + "\n"

        # Add the final subchunk if it exists
        if current_subchunk:
            sub_chunks.append(current_subchunk[:-1])
            new_labels.append(label)
            chunk_indexes.append(chunk_index)

    return sub_chunks, new_labels, chunk_indexes


def extract_text(input_path, config):
    """Extracts text from a PDF and performs basic chunking.

    Args:
        input_path (str): Path to the PDF file.
        config (dict): Configuration dictionary with processing parameters.

    Returns:
        tuple: A tuple containing:
            - list: A list of text chunks (str).
            - list: A list of labels, all "text".
    """
    extracted_texts = []
    reader = PdfReader(input_path)
    for page in reader.pages:
        extracted_texts.append(page.extract_text())

    all_chunks, all_labels = basic_chunking(extracted_texts, config)

    return all_chunks, all_labels


def extract_multimodal(model, input_path):
    """Extracts multimodal content (text, tables, figures) from PDF pages.

    Args:
        model: The multimodal extraction model (e.g., GeminiFlash).
        input_path (str): Path to the PDF file.

    Returns:
        list: A list of extracted content strings.
    """
    pages = convert_from_path(input_path)
    results = []
    query = """
    Extract text data from this image of a PDF page in Markdown format.
    Extract only the text, without saying anything else or giving any
    further explanation. Be exhaustive, extract all text information.
    Do not introduce the output with something like "Here is the extracted
    text" or similar, but reply directly.
    Label "blocks" and separate each block with a "|||" delimiter.
    A block is a part of the text that should not be divided for better
    understanding (paragraph, table...).

    Figures:
    Extract the text of the image without describing it.
    If there is no text data, return nothing. If there is a diagram, try
    to read the values of the diagram (e.g., bar values) from the axes
    and link them to the legend.

    Tables:
    IMPORTANT: Convert tables to Markdown format.
    Include the title of the text. Maintain the structure of the table
    with headers and try to associate each cell with the correct row or
    column, even if the table rows are implicit and not directly displayed.
    """

    for page in pages:
        results.append(model.predict_image(query, page, []))

    return results


def extract_chunks(input_path, config):
    """Extracts and chunks content from a file based on configuration.

    Args:
        input_path (str): Path to the input file.
        config (dict): Configuration dictionary for processing.

    Returns:
        tuple: A tuple containing:
            - list: List of extracted chunks.
            - list: List of labels for each chunk.
            - list or None: List of parent chunk indexes for sub-chunks,
                            or None if not sub-chunking.
    """
    print(f"Processing file {input_path}")
    if config["processing"]["MULTIMODAL_EXTRACTION"]:
        extraction_model = GeminiFlash(api_key=GOOGLE_API_KEY)
        text_results = extract_multimodal(extraction_model, input_path)
        config["processing"]["SEPARATOR"] = "|||"
        chunks, labels = basic_chunking(text_results, config)

    else:
        chunks, labels = extract_text(input_path, config)

    if config["processing"]["SUB_CHUNKING"]:
        chunks, labels, indexes = sub_chunking(chunks, labels)
    else:
        indexes = None

    return chunks, labels, indexes


def process(collection, file_path, config):
    """Processes a file by extracting chunks and adding them to a collection.

    Args:
        collection: The ChromaDB collection object.
        file_path (str): Path to the file to process.
        config (dict): Configuration dictionary for processing.
    """
    chunks, labels, indexes = extract_chunks(file_path, config)

    filename = os.path.basename(file_path)
    doc = collection.get()
    start_id = max([int(id[2:]) + 1 for id in doc["ids"]], default=0)

    if indexes:
        start_index = (
            max([m["chunk"] for m in doc["metadatas"] if "chunk" in m], default=-1) + 1
        )
        metadatas = [
            {"from": filename, "type": labels[j], "chunk": start_index + indexes[j]}
            for j in range(len(labels))
        ]
    else:
        metadatas = [{"from": filename, "type": labels[j]} for j in range(len(labels))]

    collection.add(
        documents=chunks,
        ids=[f"id{start_id + i}" for i in range(len(chunks))],
        metadatas=metadatas,
    )