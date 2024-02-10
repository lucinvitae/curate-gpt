import functools
import json
import logging
import os
from typing import Any, List, Optional, Union

import backoff
import chromadb
import dsp
import dspy
import openai
from chromadb import ClientAPI
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory, cache_turn_on
from dsp.modules.lm import LM
from dsp.utils import dotdict
from openai import OpenAI
from openai.lib.azure import AzureOpenAI
from openai.types.chat import ChatCompletion

from curate_gpt.utils.azure import USE_AZURE, get_azure_settings

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(message)s", handlers=[logging.FileHandler("openai_usage.log")]
)
logger = logging.getLogger(__name__)

ERRORS = (openai.RateLimitError, openai.APIError)
OpenAIObject = dict


def backoff_handler(details: dict) -> None:
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details)
    )


_client_registry = {}
CLIENT_REGISTRY_KEY = "registry_key"


def get_registry_key(**kwargs) -> str:
    return kwargs[CLIENT_REGISTRY_KEY]


@CacheMemory.cache
def v1_cached_gpt_turbo_request_v2(**kwargs) -> ChatCompletion:
    client = _client_registry[get_registry_key(**kwargs)]
    if "stringify_request" in kwargs:
        kwargs = json.loads(kwargs["stringify_request"])
    kwargs = {key: kwargs[key] for key in kwargs if key != CLIENT_REGISTRY_KEY}
    return client.chat.completions.create(**kwargs)


@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def v1_cached_gpt_turbo_request_v2_wrapped(**kwargs) -> ChatCompletion:
    return v1_cached_gpt_turbo_request_v2(**kwargs)


def chat_request(**kwargs) -> dict:
    return v1_cached_gpt_turbo_request_v2_wrapped(**kwargs).model_dump()


Client = Union[OpenAI, AzureOpenAI]


def get_openai_client(use_azure: bool, registry_key, **kwargs) -> Client:
    client_class = AzureOpenAI if use_azure else OpenAI
    if use_azure:
        config = get_azure_settings()["chat_model"]
        kwargs.update(
            {
                "api_version": config["api_version"],
                "azure_endpoint": config["base_url"],
                "api_key": config["api_key"],
                "azure_deployment": config["deployment_name"],
            }
        )
    client = client_class(**kwargs)
    _client_registry[registry_key] = client
    return client


def set_openai_client(
    use_azure: bool, registry_key: str, client_settings: dict | None = None
) -> Client:
    kwargs = {} if client_settings is None else client_settings
    return get_openai_client(use_azure, registry_key, **kwargs)


class GPT(LM):
    """Wrapper around OpenAI's GPT API. Supports both the OpenAI and Azure APIs.

    Args:
        model (str, optional): OpenAI or Azure supported LLM model to use. Defaults to "gpt-4".
        api_key (Optional[str], optional): API provider Authentication token. use Defaults to None.
        api_provider (Literal["openai", "azure"], optional): The API provider to use. Defaults to "openai".
        **kwargs: Additional arguments to pass to the API provider.
    """

    def __init__(
        self,
        model: str = "gpt-4",
        use_azure: bool = USE_AZURE,
        client_settings: dict | None = None,
        **kwargs,
    ):
        super().__init__(model)
        self.provider = "openai"

        self.registry_key = model
        self._client = set_openai_client(use_azure, self.registry_key, client_settings)

        self.kwargs = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            **kwargs,
        }  # TODO: add kwargs above for </s>

        self.history: list[dict[str, Any]] = []

    def _openai_client(self):
        return openai

    def log_usage(self, response):
        """Log the total tokens from the OpenAI API response."""
        usage_data = response.get("usage")
        if usage_data:
            total_tokens = usage_data.get("total_tokens")
            logging.info(f"{total_tokens}")

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}
        # caching mechanism requires hashable kwargs
        kwargs["messages"] = [{"role": "user", "content": prompt}]
        kwargs = {"stringify_request": json.dumps(kwargs), CLIENT_REGISTRY_KEY: self.registry_key}
        response = chat_request(**kwargs)

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_tries=3,
        max_time=1000,
        on_backoff=backoff_handler,
    )
    def request(self, prompt: str, **kwargs):
        """Handles retrieval of GPT completions whilst handling rate limiting and caching."""
        return self.basic_request(prompt, **kwargs)

    @staticmethod
    def _get_choice_text(choice: dict[str, Any]) -> str:
        return choice["message"]["content"]

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[str]:
        """Retrieves completions from GPT.

        Args:
            prompt (str): prompt to send to GPT
            only_completed (bool, optional): return only completed responses and ignores completion due to length. Defaults to True.
            return_sorted (bool, optional): sort the completion choices using the returned probabilities. Defaults to False.

        Returns:
            list of completion choices
        """

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        response = self.request(prompt, **kwargs)

        if dsp.settings.log_openai_usage:
            self.log_usage(response)

        choices = response["choices"]

        completed_choices = [c for c in choices if c["finish_reason"] != "length"]

        if only_completed and len(completed_choices):
            choices = completed_choices

        completions = [self._get_choice_text(c) for c in choices]
        if return_sorted and kwargs.get("n", 1) > 1:
            scored_completions = []

            for c in choices:
                tokens, logprobs = (
                    c["logprobs"]["tokens"],
                    c["logprobs"]["token_logprobs"],
                )

                if "<|endoftext|>" in tokens:
                    index = tokens.index("<|endoftext|>") + 1
                    tokens, logprobs = tokens[:index], logprobs[:index]

                avglog = sum(logprobs) / len(logprobs)
                scored_completions.append((avglog, self._get_choice_text(c)))

            scored_completions = sorted(scored_completions, reverse=True)
            completions = [c for _, c in scored_completions]

        return completions


class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


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


def get_openai_embedding_function(
    use_azure: bool = USE_AZURE, model_name: str = "text-embedding-ada-002"
) -> embedding_functions.OpenAIEmbeddingFunction:
    if use_azure:
        config = get_azure_settings()["embedding_model"]
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=config["api_key"],
            model_name=config.get("model_name", model_name),
            api_base=config["base_url"],
            api_type="azure",
            api_version=config["api_version"],
            deployment_id=config["deployment_name"],
        )
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name=model_name,
    )


class ChromadbForAzureRM(dspy.Retrieve):
    """
    A retrieval module that uses chromadb to return the top passages for a given query.
    This features some additional changes for successful Azure integration with Azure OpenAI
    Services deployments where the deployment identifiers differ from the model names.

    Assumes that the chromadb index has been created and populated with the following metadata:
        - documents: The text of the passage

    Args:
        collection_name (str): chromadb collection name
        persist_directory (str): chromadb persist directory
        openai_embed_model (str, optional): The OpenAI embedding model to use. Defaults to "text-embedding-ada-002".
        openai_api_key (str, optional): The API key for OpenAI. Defaults to None.
        openai_org (str, optional): The organization for OpenAI. Defaults to None.
        k (int, optional): The number of top passages to retrieve. Defaults to 3.

    Returns:
        dspy.Prediction: An object containing the retrieved passages.

    Examples:
        Below is a code snippet that shows how to use this as the default retriever:
        ```python
        llm = dspy.OpenAI(model="gpt-3.5-turbo")
        retriever_model = ChromadbRM('collection_name', 'db_path')
        dspy.settings.configure(lm=llm, rm=retriever_model)
        # to test the retriever with "my query"
        retriever_model("my query")
        ```

        Below is a code snippet that shows how to use this in the forward() function of a module
        ```python
        self.retrieve = ChromadbRM('collection_name', 'db_path', k=num_passages)
        ```
    """

    def __init__(
        self,
        client: ClientAPI,
        collection_name: str,
        model: str = "text-embedding-ada-002",
        k: int = 5,
    ):
        self.model = model
        self.client = client
        self.collection_name = collection_name
        self.openai_ef = get_openai_embedding_function(model_name=model)
        super().__init__(k=k)

    @classmethod
    def from_dir(
        cls, persist_directory: str, collection_name: str, **kwargs
    ) -> "ChromadbForAzureRM":
        client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=Settings(allow_reset=True, anonymized_telemetry=False),
        )
        return cls(client, collection_name, **kwargs)

    # @backoff.on_exception(
    #     backoff.expo,
    #     ERRORS,
    #     max_time=15,
    # )
    # def _get_embeddings(self, queries: List[str]) -> List[List[float]]:
    #     """Return query vector after creating embedding using OpenAI
    #
    #     Args:
    #         queries (list): List of query strings to embed.
    #
    #     Returns:
    #         List[List[float]]: List of embeddings corresponding to each query.
    #     """
    #
    #     model_arg = {"engine": self.model_name,
    #         "deployment_id": self.model_name,
    #         "api_version": self.api_version,
    #         "api_base": self.api_base,
    #     }
    #     embedding = self.openai_ef._client.create(
    #         input=queries, model=self._model, **model_arg,
    #     )
    #     return [embedding.embedding for embedding in embedding.data]

    def forward(
        self, query_or_queries: Union[str, List[str]], k: Optional[int] = None, **kwargs
    ) -> list[dict]:
        """Search with db for self.k top passages for query

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.

        Returns:
            An object containing the retrieved passages.
        """
        collection = self.collection_name
        logger.info(f"Searching for {query_or_queries} in {collection}")
        if isinstance(query_or_queries, list):
            raise NotImplementedError
        text = query_or_queries

        include = ["metadatas", "documents", "distances"]
        client = self.client

        # Note: with chromadb it is necessary to get the collection again;
        # the first time we do not know the embedding function, but do not
        # want to accidentally set it
        collection = client.get_collection(name=collection)

        metadata = collection.metadata
        collection = client.get_collection(name=collection.name, embedding_function=self.openai_ef)

        logger.debug(f"Collection metadata: {metadata}")
        if text:
            query_texts = [text]
        else:
            # TODO: use get()
            query_texts = ["any"]
        if k is not None:
            kwargs["n_results"] = k
        logger.debug(f"Query texts: {query_texts} include: {include}, kwargs={kwargs}")
        if query_texts == ["any"] and "n_results" not in kwargs and False:
            results = collection.get(include=include, **kwargs)
        else:
            results = collection.query(query_texts=query_texts, include=include, **kwargs)
        metadatas = results["metadatas"][0]
        results["distances"][0]
        documents = results["documents"][0]
        if "embeddings" in include:
            embeddings = results["embeddings"][0]
        else:
            embeddings = None

        results = []
        for i in range(0, len(documents)):
            if embeddings:
                embeddings[i]
            else:
                pass
            if not metadatas[i]:
                logger.error(
                    f"Empty metadata for item {i} [num: {len(metadatas)}] doc: {documents[i]}"
                )
                continue
            # meta = json.loads(metadatas[i]["_json"])
            # results.append(dotdict({"long_text": documents[i], "metadata": meta, "distance": distances[i]}))
            results.append(dotdict({"long_text": documents[i]}))

        return results


def retrieve_dspy(query: str, path: str, collection: str, retrieve_k: int = 5):
    retrieve_model = ChromadbForAzureRM.from_dir(
        persist_directory=path,
        collection_name=collection,
    )
    dspy.settings.configure(rm=retrieve_model)
    retrieve = dspy.Retrieve(k=retrieve_k)
    top_passages = retrieve(query).passages
    return top_passages


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


def rag_dspy(query: str, path: str, collection: str):
    language_model = GPT(temperature=0.7)

    dspy.settings.configure(lm=language_model)

    # Define the predictor.
    generate_answer = dspy.Predict(BasicQA)

    # Call the predictor on a particular input.
    print(f"Question: {query}")
    result = generate_answer(question=query)

    language_model.inspect_history(n=1)

    return result
