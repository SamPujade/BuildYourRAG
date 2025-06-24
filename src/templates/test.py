
def comparison_template(question: str, output: str, answer: str) -> str:
    return f"""
        Evaluate the response of a question, by comparing it to the reference answer.
        The reference answer is the correct answer to the question.
        Evaluate the response with an integer score between 0 and 2.

        Question : {question}
        Response to evaluate : {output}
        Reference answer : {answer}

        Score Rubrics:
        [Is the response correct, accurate, and factual based on the reference answer?]
        Score 0: The response is completely incorrect and/or inaccurate.
        Score 1: The response is somewhat correct and/or accurate.
        Score 2: The response is completely correct and accurate.

        Start your output with "Score: {{score}}".
"""

def comparison_template_sparkasse(question: str, output: str, answer: str) -> str:
    return f"""
        Evaluate the response of a question, by comparing it to the reference answer.
        The reference answer is the correct answer to the question.
        Evaluate the response with an integer score between 0 and 2.

        Question : {question}
        Response to evaluate : {output}
        Reference answer : {answer}

        Score Rubrics:
        [Is the response correct, accurate, and factual based on the reference answer?]
        Score 0: The response is completely incorrect and/or inaccurate.
        Score 1: The response is correct but with the wrong or uncomplete explanation.
        Score 2: The response is completely correct, with the right explanation.

        Start your output with "Score: {{score}}".
"""