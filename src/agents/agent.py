import sys
import yaml

sys.path.append("./src/")

from database.doc_processing import process
from rag import Retriever, Generator, Router


class Agent:
    def __init__(self, config):
        with open(config["agent"]["YAML_PATH"], "r", encoding="utf-8") as file:
            agent_config = yaml.safe_load(file)

        self.initial_message = agent_config["initial_message"]
        self.prompt_template = agent_config["prompt_template"]
        self.router_template = agent_config["router_template"]
        self.collection_name = agent_config["collection_name"]

        self.processing_function = process

        self.retriever = Retriever(self.collection_name, config)
        self.generator = Generator(config, template=self.prompt_template)
        self.router = Router(template=self.router_template)

    def update_collection(self, collection_name, config):
        if collection_name != self.collection_name:
            self.collection_name = collection_name
            self.retriever = Retriever(collection_name, config)

    def predict(self, model, message, history=[], user_context=[]):
        router_output = self.router.route_and_reformulate(model, message)
        print("=== Router Output ===\n", router_output, "\n")
        user_information = router_output["user_information"]
        user_context.append(user_information)

        if router_output["classification"] == "Context":
            context = self.retriever.retrieve(router_output["new_query"])
            # for c in context:
            #     print(f"\n{c}\n---------\n")
            output = self.generator.predict(
                model, message, history, context, user_context
            )
            # return self.filter_output(output, context)

        else:
            context = []
            output = self.generator.predict(model, message, history)

        print("=== Generator Output ===\n", output, "\n")
        return output, context, user_information
