"""
DSPy Pipelines for automatic prompt engineering and RAG.
"""

import logging

import dspy

from curate_gpt.pipeline.gpt import GPT
from curate_gpt.pipeline.retrieval import ChromadbForAzureRM

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(message)s", handlers=[logging.FileHandler("openai_usage.log")]
)


class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class RAG(dspy.Module):
    def __init__(self, retrieve_k=5):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=retrieve_k)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


# class RAG(dspy.Module):
#     def __init__(self, num_passages=5):
#         super().__init__()
#
#         # declare three modules: the retriever, a query generator, and an answer generator
#         self.retrieve = dspy.Retrieve(k=num_passages)
#         self.generate_query = dspy.ChainOfThought("question -> search_query")
#         self.generate_answer = dspy.ChainOfThought("context, question -> answer")
#
#     def forward(self, question):
#         # generate a search query from the question, and use it to retrieve passages
#         search_query = self.generate_query(question=question).search_query
#         passages = self.retrieve(search_query).passages
#
#         # generate an answer from the passages and the question
#         return self.generate_answer(context=passages, question=question)


def basic_qa_dspy(query: str):
    language_model = GPT(temperature=0.7)

    dspy.settings.configure(lm=language_model)

    # Define the predictor.
    generate_answer = dspy.Predict(BasicQA)

    # Call the predictor on a particular input.
    print(f"Question: {query}")
    result = generate_answer(question=query)

    language_model.inspect_history(n=1)

    return result


def retrieve_dspy(query: str, path: str, collection: str, retrieve_k: int = 5):
    retrieve_model = ChromadbForAzureRM.from_dir(
        persist_directory=path,
        collection_name=collection,
    )
    dspy.settings.configure(rm=retrieve_model)
    retrieve = dspy.Retrieve(k=retrieve_k)
    top_passages = retrieve(query).passages
    return top_passages


def rag_dspy(query: str, path: str, collection: str):
    language_model = GPT(temperature=0.7)
    retrieve_model = ChromadbForAzureRM.from_dir(
        persist_directory=path,
        collection_name=collection,
    )
    dspy.settings.configure(rm=retrieve_model)

    dspy.settings.configure(lm=language_model)

    # Define the predictor.
    generate_answer = dspy.Predict(BasicQA)

    # Call the predictor on a particular input.
    print(f"Question: {query}")
    result = generate_answer(question=query)

    language_model.inspect_history(n=1)

    return result
