from collections import defaultdict


def fill_template(template: str, **kwargs):
    """Safely fill a template with named parameters, defaulting missing ones to empty string."""
    return template.format_map(defaultdict(str, kwargs))


class Generator:
    def __init__(self, config, template):
        self.config = config
        self.prompt_template = template

    def get_input(self, query, context):
        return fill_template(self.prompt_template, message=query, context=context)

    def predict(self, model, message, history, context=None, user_context=None):
        model.change_config(self.config)

        input_text = fill_template(
            self.prompt_template,
            message=message,
            context=context or "",
            history=history or "",
            user_context=user_context or "",
        )

        return model.predict(input_text, history)
