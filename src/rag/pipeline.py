import time
import json
import re
import os
import yaml

import chromadb

from rag.embedding_model import get_model
from rag.generation_model import get_model_by_name
from templates.answering import prompt_template, general_prompt_template
from templates.router import router_template, route_and_reformulate_KID


class Router:
    def __init__(self, template=router_template):
        self.template = template

    def clean_output(self, output):
        return output.strip().strip("`")
    
    def query_routing(self, model, query):
        input = self.template(query)
        output = model.predict_json(input).strip()
        return self.clean_output(output)
    
    def route_and_reformulate(self, model, query):
        input = self.template(query)
        output = model.predict_json(input).strip()
        try:
            return json.loads(output)
        except:
            return output


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

        context, indexes  = [], []
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


class Generator:
    def __init__(self, config, template=prompt_template):
        self.config = config
        self.prompt_template = template

    def get_input(self, query, context):
        return self.prompt_template(query, context)

    def predict(self, model, message, history, *args):
        model.change_config(self.config)

        if args:
            input = self.prompt_template(message, *args)
        else:
            input = general_prompt_template(message)

        return model.predict(input, history)


class RAGPipeline:
    def __init__(self, collection_name, model=None, config=None, template=None):
        if config:
            self.config = config
        else:
            with open("src/configs/config.yaml", "r") as config_file:
                self.config = yaml.safe_load(config_file)

        self.model = model or get_model_by_name(config["generation"]["LLM"])
        self.retriever = Retriever(collection_name, self.config)
        self.generator = Generator(self.config, template=template)
        self.router = Router(template=route_and_reformulate_KID)

    def retrieve(self, query_text):
        return self.retriever.retrieve(query_text)

    def query(self, query_text):
        t0 = time.time()

        result = self.retriever.retrieve(query_text)
        context = result["documents"][0]
        t1 = time.time()
        print(f"Retrieval time : {t1 - t0}")

        output = self.generator.chat_completion(query_text, context)
        print(f"Generation time : {time.time() - t1}")
        return output, context

    def filter_output(self, output, context):
        try:
            output_json = json.loads(output)
        except:
            match = re.search(r"\{.*?\}", output)
            if match:
                try:
                    output_json = json.loads(match.group())
                except:
                    return output, []
            else:
                return output, []

        try:
            used_context = [context[i] for i in output_json["Context"]]
            return output_json["Answer"], used_context
        except:
            try:
                return output_json["Answer"], []
            except:
                return output, []

    def predict(self, model, message, history=[]):
        router_output = self.router.route_and_reformulate(model, message)

        if router_output["classification"] == "Context":
            context = self.retriever.retrieve(router_output["new_query"])
            # for c in context:
            #     print(f"\n{c}\n---------\n")
            output = self.generator.predict(message, history, context)
            return self.filter_output(output, context)

        else:
            output = self.generator.predict(model, message, history, [])
            return output, []

