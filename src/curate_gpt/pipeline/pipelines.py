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


class PredictHPOs(dspy.Signature):
    __doc__ = """Given a snippet from a patient's medical history, identify the Human Phenotype Ontology (HPO) identifier for each phenotype in the text. If none are mentioned in the snippet, say '\n'."""

    context = dspy.InputField()
    hpo_ids = dspy.OutputField(
        desc="list of comma-separated HPO IDs",
        format=lambda x: ", ".join(x) if isinstance(x, list) else x,
    )


class CoT(dspy.Module):
    def __init__(self):
        super().__init__()

        self.generate_answer = dspy.ChainOfThought(PredictHPOs)

    def forward(self, context, labels=None):
        return self.generate_answer(context=context)


class SearchQueryForHPOs(dspy.Signature):
    __doc__ = """Given a snippet from a patient's medical history, create a search query for the Human Phenotype Ontology (HPO) identifier for each phenotype in the text."""

    context = dspy.InputField()
    search_query = dspy.OutputField(desc="search query to retrieve HPO document texts")


class PredictWithSearchHPOs(dspy.Signature):
    __doc__ = """Given a snippet from a patient's medical history and the search results, identify the Human Phenotype Ontology (HPO) identifier for each phenotype in the text. If none are mentioned in the snippet, say '\n'."""

    context = dspy.InputField()
    documents = dspy.InputField(
        desc="HPO document texts", format=lambda x: "\n\n".join(x) if isinstance(x, list) else x
    )
    hpo_ids = dspy.OutputField(
        desc="list of comma-separated HPO IDs",
        format=lambda x: ", ".join(x) if isinstance(x, list) else x,
    )


class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        # declare three modules: the retriever, a query generator, and an answer generator
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_query = dspy.ChainOfThought(SearchQueryForHPOs)
        self.generate_answer = dspy.ChainOfThought(PredictWithSearchHPOs)

    def forward(self, context, labels=None):
        # generate a search query from the context, and use it to retrieve passages
        search_query = self.generate_query(context=context).search_query
        documents = self.retrieve(search_query).passages

        # generate an answer from the passages and the question
        return self.generate_answer(context=context, documents=documents)


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
