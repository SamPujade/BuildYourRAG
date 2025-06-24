def prompt_template(query: str, context: list) -> str:
    """
    Generate a prompt for text generation from context.
    Args:
        - query (str): question or prompt to be answered
        - context (list): optional additional information for the task
    Returns
        - str: formatted string containing either just the query or both the query and context
    """
    if context:
        return f"""
            Beantworten Sie die Frage nur auf der Grundlage des folgenden Kontextes:
            {context}
            Beantworten Sie die Frage auf der Grundlage des oben genannten Kontextes: {query}.
            Geben Sie eine detaillierte Antwort.
            Beantworten Sie auf Deutsch.
            Begründen Sie Ihre Antworten nicht.
            Geben Sie keine Informationen an, die nicht in den Kontext erwähnt werden.
            Sagen Sie nicht "gemäß dem Kontext", "im Kontext erwähnt", "Auswug aus dem Kontext" oder etwas Ähnliches.
            Wenn der Kontext nicht genügend Informationen liefert, sagen Sie einfach, dass Sie es nicht wissen, und versuchen Sie nicht, sich eine Antwort auszudenken.
        """
        # Geben Sie das verwendete Element aus dem Kontext, in dem die Informationen enthalten sind.
        # Antworten Sie in folgendem JSON Format:
        # {{ "Context": "verwendetes Kontextelement", "Answer": "[Antwort]" }}
    else:
        return query


def prompt_template_english(query: str, context: list) -> str:
    """
    Generate a prompt for text generation from context.
    Args:
        - query (str): question or prompt to be answered
        - context (list): optional additional information for the task
    Returns
        - str: formatted string containing either just the query or both the query and context
    """
    if context:
        return f"""
            Answer the question based on the following context list only:
            {context}
            Answer the question based on the above context: {query}.
            Give a detailed answer.
            Do not justify your answers.
            Do not give information that is not mentioned in the context.
            Do not say “according to the context”, “mentioned in the context”, “extracted from the context” or anything similar.
            If the context does not provide enough information, simply say that you do not know and do not try to think of an answer.
            Provide the indices of the elements used from the context list for finding the information.
            Answer in the following JSON format:
            {{ “Context”: “[list of context element indices]", “Answer”: “[answer]” }}
        """
    else:
        return query


def prompt_template_2(query: str, context: str) -> str:
    """
    Generate a prompt for text generation from context.
    Args:
        - query (str): question or prompt to be answered
        - context (str): optional additional information for the task
    Returns
        - str: formatted string containing either just the query or both the query and context
    """
    return f"""
        Beantworten Sie die Frage : {query}.
        Geben Sie eine detaillierte Antwort.
        Beantworten Sie auf Deutsch.
        Begründen Sie Ihre Antworten nicht.
        """


def general_prompt_template(query: str) -> str:
    """
    Generate a prompt for text generation without context.
    Args:
        - query (str): question or prompt to be answered
    Returns
        - str: formatted string containing the input.
    """
    return f"""
        Du bist ein KI-Assistent für Finanzen. Deine Hauptaufgabe ist es, Fragen zu Finanzinvestitionen in ETFs zu beantworten.

        Wenn die Frage des Benutzers zu allgemein ist, antworte mit: "Ich kann nur Fragen zu Finanzinvestitionen in ETFs beantworten. Bitte stelle eine spezifische Frage zu ETFs."

        Wenn die Frage spezifisch und relevant für Finanzinvestitionen ist, antworte so genau und präzise wie möglich. Wenn die Frage vage ist, bitte den Benutzer um Klärung.

        Frage : {query}
    """



def prompt_template_KID(query: str, context: list, names: list) -> str:
    """
    Generate a prompt for text generation from context.
    Args:
        - query (str): question or prompt to be answered
        - context (list): top chunks retrieved
    """
    if context:
        return f"""
            **Instructions:**  Answer the following question by using a set of Key Information Documents (KIDs) for financial products, which are provided below as "Context".  Use this context to answer the user's question accurately and concisely.  If the context does not contain the answer, say "I cannot answer this question based on the provided information."  Always prioritize information found within the provided context.  Do not hallucinate or invent information.

            **Available Products:** {names}

            **User Query:** {query}

            **Context:** {context}
        """
    else:
        return query


def prompt_template_KID_german(query: str, context: list, user_context: str) -> str:
    """
    Generate a prompt for text generation from context.
    Args:
        - query (str): question or prompt to be answered
        - context (list): top chunks retrieved
    """
    if context:
        return f"""
            **Anleitungen:**  
            Beantworten Sie die folgende Frage von einem Benutzer mit Hilfe einer Reihe von Key Information Documents (KIDs) für Finanzprodukte, die unten als „Kontext“ angegeben sind.
            Verwenden Sie diesen Kontext, um die Frage des Benutzers genau und mit einer Erklärung zu beantworten.
            Wenn der Kontext die Antwort nicht enthält, sagen Sie: „Diese Frage kann ich anhand der mir vorliegenden Informationen nicht beantworten“.
            Wenn Sie die Frage beantworten können, aber nur mit eingen Benutzerinformationen, fragen Sie danach.
            Beantworten Sie aber vorrangig die Frage, ohne um eine Klärung zu bitten
            Priorisieren Sie immer die Informationen, die Sie im angegebenen Kontext finden.
            Wenn sich die Antwort auf ein oder mehrere Produkte bezieht, gib deren Namen an.
            Sie können die Benutzerinformationen für eine bessere Antwort verwenden.
            Halluzinieren Sie nicht und erfinden Sie keine Informationen.
            SEHR WICHTIG: Erwähnen Sie nicht den Kontext, die Dokumente, die vorliegenden Informationen oder dass Sie gemäß den bereitgestellten Informationen antworten. Der Benutzer darf nicht wissen, dass Sie Ihre Antwort auf den Kontext oder auf bestimmte Dokumente gestützt haben.

            ** Frage:** {query}

            ** Benutzerinformationen:** {user_context}

            **Kontext:** {context}
        """
    else:
        return query