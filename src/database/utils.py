import re
import sys
import os
import json
from dotenv import load_dotenv

sys.path.append("./src/")

from models.generation import GeminiFlash


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def extract_product_name(texts):
    patterns = [
        r"(?<=Produkt:\s)([^\n]*)",
        r"(?<=Ziele: Der )(.*?)(?= ist)",
        r"(?<=Name des Produkts : )(.*?)(?= \()",
        r"(?<=Name des Produkts : )(.*?)(?=\n\()",
    ]

    entry = "\n".join(texts)
    for pattern in patterns:
        match = re.search(pattern, entry)
        if match:
            return match.group(1)

    print("No product name detected.")
    print(repr(texts[0])[:1000])
    return None


def extract_risk_level(texts):
    pattern = re.compile(r"Risikoklasse\s([1-7])")

    for text in texts:
        match = pattern.search(text)
        if match:
            return int(match.group(1))

    raise Exception("No risk class found.")


def entity_extraction(input, template):
    model = GeminiFlash(api_key=GOOGLE_API_KEY, response_format="application/json")
    prompt_input = template(input)
    output = json.loads(model.predict(prompt_input))

    for key in output:
        if output[key] is None:
            print(f"Warning - no entity detected: {key}")
            output[key] = ""

    return output
