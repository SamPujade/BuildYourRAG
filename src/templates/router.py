def router_template(query):
    return f"""
    Given the user question below, classify it as either `Context`, or `Other`.
    `Context` means that a context needs to be provided for answering to question.
    `Other` is just a general question without any context needed.

    Do not respond with more than one word.

    <question>
    {query}
    </question>

    Classification:"""


def reformulate_query(query: str):
    return f"""
        I am using a RAG system on ETF financial products (description documents). I have the following query : 

        {query}

        Write a new query based on this query that will be used to retrieve context. The new generated query will be embedded by an embedding model, and then copmared to every element of my vector database using similarity search. Write the best query to capture the semantic meaning and retrieve the correct chunks.
    """


def route_and_reformulate_KID(query):
    return f"""
    You are a router for a RAG system based on ETF financial products (description documents).
    Your task is to understand the user query, classify it, reformulate it acccording to the classification found, and extract any user information from the query.

    - Step 1

    Given the user question below, classify it as either `Context`, or `Other`.
    `Context` means that a context needs to be provided for answering to question.
    The context needed is taken from a set of key information documents of ETFs, describing the products from manufacturers.
    `Other` is just a general question without any context needed from ETFs. Return `Other` if you don't need the information document of one or more ETFs for answering the question.

    <question>
    {query}
    </question>

    classification: <`Context` or `Other`>

    - Step 2

    If the classification is `Context`, write a new query based on this query that will be used to retrieve context and generate a response. 
    The new generated query will be embedded by an embedding model, and then compared to every element of the vector database using similarity search. 
    Write the best query to capture the semantic meaning, retrieve the correct chunks and generate a clear answer.
    The new query must be more specific, detailed, and likely to retrieve relevant information.

    If the classification is `Other`, simply return the original query.

    new_query: <new query>

    - Step 3

    Extract any information on the user that can be useful for giving a financial advice (like age, income, investment objective, investment horizon, risk attitude, savings plan, diversification...)
    If there isn't such information in the query, simply return an empty string "".

    user_information: <user_information>

    - Final output: JSON format dictionary with keys "classification", "new_query" and "user_information".

    """
