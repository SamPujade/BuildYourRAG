import sys
import yaml

sys.path.append("./src/")

from database.doc_processing import process_KID
from rag.pipeline import Retriever, Generator, Router
from templates.answering import prompt_template_KID_german
from templates.router import route_and_reformulate_KID

class Agent:
    def __init__(self,
            initial_message,
            router_template,
            prompt_template,
            processing_function,
            collection_name,
            config=None
        ):
        if not config:
            with open("src/configs/config.yaml", "r") as config_file:
                config = yaml.safe_load(config_file)

        self.initial_message = initial_message
        self.prompt_template = prompt_template
        self.processing_function = processing_function
        self.collection_name = collection_name
        self.retriever = Retriever(collection_name, config)
        self.generator = Generator(config, template=prompt_template)
        self.router = Router(template=router_template)

    def update_collection(self, collection_name, config):
        if collection_name != self.collection_name:
            self.collection_name = collection_name
            self.retriever = Retriever(collection_name, config)


class KIDAgent(Agent):
    def __init__(self):
        super().__init__(
            initial_message="Hallo, ich bin dein Finanz Copilot und berate dich gerne zu allem rund um das Thema Finanzen, Kapitalmarkt, Geldanlage, Rente, Versicherungen und insgesamt zu allen Bankprodukten, die wir anbieten. Du kannst mir deine Fragen stellen und ich werde dir auf Basis meiner Daten Informationen bereitstellen. Bitte nimm diese nicht als Anlageberatung wahr, sondern überprüfe die Infos nochmals gesondert.",
            router_template=route_and_reformulate_KID,
            prompt_template=prompt_template_KID_german,
            processing_function=process_KID,
            collection_name="KID_3",
        )

    def predict(self, model, message, history=[], user_context=[]):
        router_output = self.router.route_and_reformulate(model, message)
        print(router_output)
        user_information = router_output["user_information"]
        user_context.append(user_information)

        if router_output["classification"] == "Context":
            context = self.retriever.retrieve(router_output["new_query"])
            # for c in context:
            #     print(f"\n{c}\n---------\n")
            output = self.generator.predict(model, message, history, context, user_context)
            # return self.filter_output(output, context)
            return output, [], user_information

        else:
            output = self.generator.predict(model, message, history)
            return output, [], user_information