
def generation_template(context):
    return f"""
        Context: {context}
        Generate a question that can be answered from the given context.
        Do not create a question like "What ist the content of the context ?".
        Provide the answer of the generated question.
        Provide the type of data from the context : text or table.
        Your output must be in a JSON format with the following keys : "Question", "Answer", "Type".
    """


def generation_template_german(context):
    return f"""
        Kontext: {context}
        Erzeugen Sie eine Frage, die aus dem gegebenen Kontext beantwortet werden kann.
        Erstellen Sie keine Frage wie „Was ist der Inhalt des Kontextes?“.
        Geben Sie die Antwort auf die generierte Frage an.
        Geben Sie den Typ der Daten aus dem Kontext an: "Text" oder "Table".
        Ihre Ausgabe muss in einem JSON-Format mit den folgenden Schlüsseln erfolgen: "Question", "Answer", "Type".
    """

def fine_tuning_template(chunk):
    return f"""
        Convert this text to a list of questions and answers covering the whole text, in JSON format.

        Text: {chunk}

        Use this JSON schema:

        Q&A = {{'Question': str, 'Answer': str}}
        Return: list[Q&A]
    """

def fine_tuning_template_german(chunk):
    return f"""
        Konvertieren Sie diesen Text in eine Liste von Fragen und Antworten, die den gesamten Text umfassen, im JSON-Format. 
        Die Fragen und Antworten müssen in deutscher Sprache sein. 
        Wiederholen Sie die Frage in der Antwort.

        Text: {chunk}

        Verwenden Sie dieses JSON-Schema:

        Q&A = {{'Question': str, 'Answer': str}}
        Rückgabe: list[Q&A]

        Beispiel:
        [
        {{
            "Question: "Wie hat sich das verfügbare Kapital von ABB zwischen dem 31. März 2023 und dem 31. März 2024 entwickelt?",
            "Answer": "Das verfügbare Kapital von ABB zum 31. März 2024 betrug 2.577 Mio. €, während es zum 31. März 2023 2.482 Mio. € betrug."
        }}
        ]
    """