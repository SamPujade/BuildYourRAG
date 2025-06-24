import os
import yaml
from chromadb.utils import embedding_functions

root_dir = os.path.abspath(os.path.join(__file__, "..", ".."))

with open("src/configs/config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

DEVICE = config["hardware"]["DEVICE"]


class EmbeddingModel(object):
    """Main class for Embedding Model
    Args:
    - model_type (str): name of the model
    """

    def __init__(self):
        self.model_type = None
        self.device = DEVICE


class Multilingual(EmbeddingModel):
    def __init__(
        self,
    ):
        super().__init__()
        self.model_type = "Multilingual"
        self.model_name = "intfloat/multilingual-e5-large-instruct"
        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.model_name, device=self.device
            )
        )
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def embed(self, query):
        return self.embedding_function([query])[0]


def get_model(name):
    model_dict = {"Multilingual": Multilingual}
    if name in model_dict:
        return model_dict[name]()
    else:
        raise Exception(f"Model {name} not found.")
