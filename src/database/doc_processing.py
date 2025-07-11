import os
import sys
from dotenv import load_dotenv

# third-party imports
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from PIL import Image


sys.path.append("./src/")

from database.utils import entity_extraction
from models.generation import GeminiFlash
from templates.processing import prompt_entity_extraction_KID

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PDF_CONFIG_PATH = os.path.abspath("src/configs/extract_kit_config.yaml")

starter_labels = {
    "figure": "Figure:\n",
    "figure_caption": "Figure caption:\n",
    "table": "Table:\n",
    "table_caption": "Table caption:\n",
    "table_footnote": "Table footnote:\n",
}

new_labels = {
    "title": "title",
    "plain text": "text",
    "figure": "figure",
    "figure_caption": "figure",
    "table": "table",
    "table_caption": "table",
    "table_footnote": "table",
    "isolate_formula": "formula",
    "formula_caption": "formula",
}


def token_len(chunk):
    return len(chunk.split())


def split_text(text, max_chunk_size, separator="\n"):
    entries = text.split(separator)
    chunks, current_chunk = [], ""

    for entry in entries:
        if token_len(current_chunk + " " + entry) > max_chunk_size:
            chunks.append(current_chunk + separator)
            current_chunk = entry
        else:
            current_chunk += entry + separator

    if current_chunk:
        chunks.append(current_chunk[:-1])

    return chunks


def basic_chunking(texts, config):
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


def recursive_chunking(texts, labels, config):
    max_chunk_size = config["processing"]["MAX_CHUNK_SIZE"]
    overlap = config["processing"]["OVERLAP"]
    separator = config["processing"]["SEPARATOR"]
    chunks, chunk_labels = [], []
    i = 0
    current_title = None

    # Split initial chunks if they are too big
    initial_chunks, initial_labels = [], []
    for k, entry in enumerate(texts):
        split_txt = split_text(entry, max_chunk_size, separator)

        if labels[k] in starter_labels:
            split_txt = [starter_labels[labels[k]] + txt for txt in split_txt]

        initial_chunks.extend(split_txt)
        initial_labels.extend(len(split_txt) * [labels[k]])

    # Recursively merge chunks
    while i < len(initial_chunks):
        current_label = initial_labels[i]
        merged_text = initial_chunks[i]
        i += 1

        # Add current title to the beginning of the chunk
        if current_label != "title" and current_title:
            merged_text = current_title + "\n" + merged_text

        # Continue merging until the maximum chunk size is reached
        while (
            i < len(initial_chunks)
            and token_len(merged_text + " " + initial_chunks[i]) < max_chunk_size
        ):
            if initial_labels[i] == "title":
                current_title = initial_chunks[i]
            elif initial_labels[i] == "table" or initial_labels[i] == "figure":
                current_label = initial_labels[i]
            merged_text += "\n" + initial_chunks[i]
            i += 1

        # Append the merged text and corresponding label
        chunks.append(merged_text)
        chunk_labels.append(current_label)

    # for i, chunk in enumerate(chunks):
    #     print(f"\n\nCHUNK {i}\n")
    #     print(chunk)

    return chunks, chunk_labels


def sub_chunking(chunks, labels):
    """
    Splits each chunk into subchunks.

    Parameters:
        chunks (list of str): List of input chunks.
        labels (list): Corresponding labels for the input chunks.

    Returns:
        sub_chunks: List of resulting subchunks.
        labels: Corresponding labels for each subchunk.
        chunk_indexes: List of indexes indicating the parent chunk of each subchunk.
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
    extracted_texts = []
    reader = PdfReader(input_path)
    for page in reader.pages:
        extracted_texts.append(page.extract_text())

    all_chunks, all_labels = basic_chunking(extracted_texts, config)

    return all_chunks, all_labels


def extract_multimodal_all(model, input_path):
    pages = convert_from_path(input_path)
    results = []
    query = """
    Extrahieren Sie die Textdaten in diesem Bild einer PDF-Seite in einem Markdown Format. 
    Extrahieren Sie nur den Text, ohne etwas anderes zu sagen oder eine weitere Erklärung zu geben. 
    Seien Sie erschöpfend, extrahieren Sie alle Textinformationen.
    Leiten Sie die Ausgabe nicht mit etwas wie „Hier ist der extrahierte Text“ oder ähnlichem ein, sondern antworten Sie direkt.
    Kennzeichnen Sie „Blöcke“ und trennen Sie jeden Block mit einem "|||" Trennzeichen ab. Ein Block ist ein Teil des Textes, der zum besseren Verständnis nicht geteilt werden sollte (Absatz, Tabelle...).


    Abbildungen :
    Extrahieren Sie den Text des Bildes, ohne es zu beschreiben. 
    Wenn es keine Textdaten gibt, geben Sie nichts zurück. Wenn es ein Diagramm gibt, versuchen Sie, die Werte des Diagramms (z. B. Balkenwerte) anhand der Achsen zu lesen und sie mit der Legende zu verknüpfen.“

    Tabellen :
    WICHTIG: Konvertieren Sie Tabellen in ein Markdown-Format. 
    Fügen Sie den Titel des Textes ein. Behalten Sie die Struktur der Tabelle mit Überschriften bei und versuchen Sie, jede Zelle der richtigen Zeile oder Spalte zuzuordnen, auch wenn die Zeilen der Tabelle implizit sind und nicht direkt angezeigt werden."""

    for page in pages:
        results.append(model.predict_image(query, page, []))

    return results


def extract_multimodal(model, images):
    # query = "Extrahieren Sie die Textdaten in diesem Bild in einem strukturierten Format. Extrahieren Sie nur den Text, ohne etwas anderes zu sagen oder weitere Erklärungen zu geben. Wenn es keine Textdaten gibt, wird nur zurückgegeben: „kein Text“."
    query = "Extrahieren Sie die Textdaten in diesem Bild in einem strukturierten Format. Extrahieren Sie nur den Text, ohne etwas anderes zu sagen oder weitere Erklärungen zu geben. Wenn es keine Textdaten gibt, wird nur zurückgegeben: „kein Text“. Wenn ein Diagramm vorhanden ist, versuchen Sie, die Werte des Diagramms (z. B. Balkenwerte) mithilfe der Achse zu lesen und mit der Legende zu verknüpfen."
    return [
        model.predict_image(query, Image.fromarray(image), []) + "\n"
        for image in images
    ]


def extract_multimodal_table(model, images):
    # query = "Extrahieren Sie die Textdaten in diesem Bild in einem strukturierten Format. Extrahieren Sie nur den Text, ohne etwas anderes zu sagen oder weitere Erklärungen zu geben. Wenn es keine Textdaten gibt, wird nur zurückgegeben: „kein Text“."
    query = "Extrahieren Sie die Textdaten in diesem Bild einer Tabelle in einem strukturierten Format. Extrahieren Sie nur den Text, ohne etwas anderes zu sagen oder eine weitere Erklärung zu geben. Konvertieren Sie die Tabelle in ein Markdown-Format. Fügen Sie den Titel des Textes ein. Behalten Sie die Struktur der Tabelle mit Überschriften bei und versuchen Sie, jede Zelle der richtigen Zeile oder Spalte zuzuordnen, auch wenn die Zeilen der Tabelle implizit sind und nicht direkt angezeigt werden."
    return [
        model.predict_image(query, Image.fromarray(image), []) + "\n"
        for image in images
    ]


def extract_chunks(input_path, config, output_path="data/outputs/layout_detection/"):
    print(f"Processing file {input_path}")
    if config["processing"]["MULTIMODAL_EXTRACTION"]:
        extraction_model = GeminiFlash(api_key=GOOGLE_API_KEY)
        text_results = extract_multimodal_all(extraction_model, input_path)
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


def process_KID(collection, file_path, config):
    chunks, labels, indexes = extract_chunks(file_path, config)

    # Extract entities and add them to metadatas
    all_chunks = "\n\n".join(chunks)
    entities = entity_extraction(all_chunks, prompt_entity_extraction_KID)
    document_metadatas = entities.copy()
    document_metadatas["from"] = os.path.basename(file_path)

    doc = collection.get()
    start_id = max([int(id[2:]) + 1 for id in doc["ids"]], default=0)

    if indexes:
        start_index = (
            max([m["chunk"] for m in doc["metadatas"] if "chunk" in m], default=-1) + 1
        )
        metadatas = [
            {
                **document_metadatas,
                "type": labels[j],
                "chunk": start_index + indexes[j],
            }
            for j in range(len(labels))
        ]
    else:
        metadatas = [
            {
                **document_metadatas,
                "type": labels[j],
            }
            for j in range(len(labels))
        ]

    collection.add(
        documents=chunks,
        ids=[f"id{start_id + i}" for i in range(len(chunks))],
        metadatas=metadatas,
    )
