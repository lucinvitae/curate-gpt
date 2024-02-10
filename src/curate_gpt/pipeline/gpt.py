import json
import logging
from typing import Any, Union

import backoff
import dsp
import openai
from dsp import LM
from openai import OpenAI
from openai.lib.azure import AzureOpenAI
from openai.types.chat import ChatCompletion

from curate_gpt.utils.azure import USE_AZURE, get_azure_settings

ERRORS = (openai.RateLimitError, openai.APIError)
OpenAIObject = dict

logging.basicConfig(
    level=logging.DEBUG, format="%(message)s", handlers=[logging.FileHandler("openai_usage.log")]
)


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


# @CacheMemory.cache
def v1_cached_gpt_turbo_request_v2(**kwargs) -> ChatCompletion:
    client = _client_registry[get_registry_key(**kwargs)]
    if "stringify_request" in kwargs:
        kwargs = json.loads(kwargs["stringify_request"])
    kwargs = {key: kwargs[key] for key in kwargs if key != CLIENT_REGISTRY_KEY}
    return client.chat.completions.create(**kwargs)


# @functools.lru_cache(maxsize=None if cache_turn_on else 0)
# @NotebookCacheMemory.cache
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
