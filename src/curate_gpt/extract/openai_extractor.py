"""Extractor that uses OpenAI functions."""
import json
import logging
from dataclasses import dataclass
from typing import List, Union

import yaml
from openai import OpenAI
from openai.lib.azure import AzureOpenAI

from curate_gpt.extract.extractor import AnnotatedObject, Extractor
from curate_gpt.utils.azure import USE_AZURE, get_azure_settings

FUNC_NAME = "extract_data"

logger = logging.getLogger(__name__)


@dataclass
class OpenAIExtractor(Extractor):

    """
    Extractor that uses OpenAI functions.
    """

    max_tokens: int = 3000
    model: str = "gpt-4"
    # conversation: List[Dict[str, Any]] = None
    # conversation_mode: bool = False

    def __init__(self, use_azure: bool = USE_AZURE):
        self._client = self.get_client(use_azure)

    def functions(self):
        return [
            {
                "name": FUNC_NAME,
                "description": "A n ontology term",
                "parameters": self.schema_proxy.json_schema(),
            },
        ]

    def extract(
        self,
        text: str,
        target_class: str,
        examples: List[AnnotatedObject] = None,
        examples_as_functions=False,
        conversation=None,
        **kwargs,
    ) -> AnnotatedObject:
        messages = [
            {
                "role": "system",
                "content": f"You are system that returns {target_class} object in JSON.",
            },
        ]
        for example in examples:
            ex_text = example.text
            ex_object = example.object
            print(f"EXAMPLE = {ex_text}")
            messages.append(
                {
                    "role": "user",
                    "content": f"make terms for {ex_text}",
                }
            )
            if not examples_as_functions:
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": FUNC_NAME,
                            "arguments": json.dumps(ex_object),
                        },
                    },
                )
            else:
                messages.append(
                    {
                        "role": "function",
                        "name": FUNC_NAME,
                        "content": json.dumps(ex_object),
                    },
                )
        if conversation:
            messages.extend(conversation)
        # content = f"make terms for {text}"
        content = text
        messages.append(
            {
                "role": "user",
                "content": content,
            }
        )
        print(yaml.dump(messages))
        response = self._client.chat.completions.create(
            model=self.model,
            functions=self.functions(),
            messages=messages,
            max_tokens=self.max_tokens,
        )
        logger.debug(f"RESPONSE = {response}")
        print(response)
        choice = response.choices[0]
        message = choice["message"]
        if "function_call" not in message:
            if self.raise_error_if_unparsable:
                raise ValueError("No function call in response")
            r = "{}"
        else:
            r = message["function_call"]["arguments"]
        try:
            obj = json.loads(r)
            if conversation:
                conversation.append(messages[-1])
                conversation.append({"role": "function", "name": FUNC_NAME, "content": r})
        except json.decoder.JSONDecodeError as e:
            if self.raise_error_if_unparsable:
                raise e
            obj = {}
        return AnnotatedObject(object=obj)

    def get_client(self, use_azure: bool, **kwargs) -> Union[OpenAI, AzureOpenAI]:
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
        return client_class(**kwargs)
