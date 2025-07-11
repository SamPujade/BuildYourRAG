import json


class Router:
    def __init__(self, template):
        self.template = template

    def clean_output(self, output):
        return output.strip().strip("`")

    def query_routing(self, model, query):
        input = self.template.format(query=query)
        output = model.predict_json(input).strip()
        return self.clean_output(output)

    def route_and_reformulate(self, model, query):
        input = self.template.format(query=query)
        output = model.predict_json(input).strip()
        try:
            return json.loads(output)
        except:
            return output
