import yaml
import os
import re
import base64

import google.generativeai as genai

root_dir = os.path.abspath(os.path.join(__file__, "..", ".."))

with open("src/configs/config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

MODEL_FOLDER = config["generation"]["MODEL_FOLDER"]
TEMPERATURE = config["generation"]["TEMPERATURE"]
MAX_TOKENS = config["generation"]["MAX_TOKENS"]
N_CTX = config["generation"]["N_CTX"]


def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:image/png;base64,{base64_data}"


class GenerationModel(object):
    """Main class for Generation Model
    Args:
    - model_type (str): name of the model
    """

    def __init__(self):
        self.model_type = None

    def change_config(self, config):
        pass


class GoogleAI(GenerationModel):
    def __init__(self, api_key, response_format):
        super().__init__()
        self.path = None
        self.response_format = response_format
        genai.configure(api_key=api_key)

    def init_model(self):
        if self.response_format:
            self.model = genai.GenerativeModel(
                self.path,
                generation_config={"response_mime_type": self.response_format},
            )
        else:
            self.model = genai.GenerativeModel(self.path)

    def predict(self, input, history=[]):
        messages = [
            {
                "role": "user",
                "parts": ["You are a helpful assistant."],
            }
        ]
        for history_message in history:
            messages.append(
                {
                    "role": history_message["role"],
                    "parts": [history_message["content"]],
                }
            )
        chat = self.model.start_chat(history=messages)
        response = chat.send_message(input)
        return response.text

    def predict_json(self, input, history=[]):
        original_response_format = self.response_format
        self.response_format = "application/json"
        self.init_model()

        output = self.predict(input, history)
        self.response_format = original_response_format
        self.init_model()

        return output

    def predict_image(self, input, image, history):
        messages = [{"role": "user", "parts": ["You are a helpful assistant."]}]
        for history_message in history:
            messages.append(
                {
                    "role": history_message["role"],
                    "parts": [history_message["content"]],
                }
            )
        chat = self.model.start_chat(history=messages)
        response = chat.send_message([input, image])
        return response.text


class GeminiFlash(GoogleAI):
    def __init__(self, api_key, response_format=None):
        super(GeminiFlash, self).__init__(api_key, response_format)
        self.path = "gemini-1.5-flash"
        self.input_token_limit = 500000
        self.init_model()


class GeminiFlash8b(GoogleAI):
    def __init__(self, api_key, response_format=None):
        super(GeminiFlash, self).__init__(api_key, response_format)
        self.path = "gemini-1.5-flash-8b"
        self.init_model()


class GeminiPro(GoogleAI):
    def __init__(self, api_key, response_format=None):
        super(GeminiFlash, self).__init__(api_key, response_format)
        self.path = "gemini-1.5-pro"
        self.init_model()


class GeminiFlash2Exp(GoogleAI):
    def __init__(self, api_key, response_format=None):
        super(GeminiFlash2Exp, self).__init__(api_key, response_format)
        self.path = "gemini-2.0-flash-exp"
        self.init_model()


model_classes = {
    "Gemini 1.5 Flash": GeminiFlash,
    "Gemini 1.5 Pro": GeminiPro,
    "Gemini 2.0 Flash Experimental": GeminiFlash2Exp,
}


def get_model_by_name(name, api_key=None):
    if name in model_classes:
        return model_classes[name](api_key=api_key)
    else:
        raise ValueError(f"No model found: {name}")


def get_model_names():
    return list(model_classes.keys())
