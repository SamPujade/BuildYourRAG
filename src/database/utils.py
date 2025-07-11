import re
import sys
import os
import json
from dotenv import load_dotenv

sys.path.append("./src/")

from models.generation import GeminiFlash


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def entity_extraction(input, template):
    model = GeminiFlash(api_key=GOOGLE_API_KEY, response_format="application/json")
    prompt_input = template(input)
    output = json.loads(model.predict(prompt_input))

    for key in output:
        if output[key] is None:
            print(f"Warning - no entity detected: {key}")
            output[key] = ""

    return output
