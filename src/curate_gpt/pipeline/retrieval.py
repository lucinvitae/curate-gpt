import logging
import os
from typing import List, Optional, Union

import chromadb
import dspy
from chromadb import ClientAPI, Settings
from chromadb.utils import embedding_functions
from dsp import dotdict

from curate_gpt.utils.azure import USE_AZURE, get_azure_settings

logger = logging.getLogger(__name__)


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
        use_azure: bool = USE_AZURE,
    ):
        self.model = model
        self.client = client
        self.collection_name = collection_name
        self.openai_ef = get_openai_embedding_function(model_name=model, use_azure=use_azure)
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
            # TODO: refactor/ add backoff/retry with @backoff.add_handler similar to gpt.py
            results = collection.get(include=include, **kwargs)
        else:
            # TODO: refactor/ add backoff/retry with @backoff.add_handler similar to gpt.py
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
