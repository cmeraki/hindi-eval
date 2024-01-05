"""
Simple wrapper class around GPT API
"""

import json
import openai
import backoff

from typing import Tuple, Dict, List
from .utils.logger import DataPrepLogger

logger = DataPrepLogger(__name__).get_logger()

class GPTGenerator():
    def __init__(self, model_id: str) -> None:
        from dotenv import load_dotenv
        load_dotenv()

        assert model_id in ['gpt-4-1106-preview', 'gpt-3.5-turbo-1106']

        self.client = openai.OpenAI()
        self.model_id = model_id

    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=300)
    def __call__(
        self,
        messages: List[str],
        json_mode: bool = True,
        **kwargs
    ) -> Tuple[Dict, any]:

        if json_mode:
            completions = self.client.chat.completions.create(
                model=self.model_id,
                response_format={'type': 'json_object'},
                messages=messages,
                **kwargs
            )

            if completions.choices[0].finish_reason == 'length':
                raise IOError(f'Reached maximum output length, output format is not reliable. {completions.choices[0].message.content}')

            op = json.loads(completions.choices[0].message.content)

        else:
            completions = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                **kwargs
            )

            op = completions.choices[0].message.content

        logger.debug(f'Prompts: {messages}, output: {op}')
        logger.debug(f'Tokens used in generation using {self.model_id}: {completions.usage}')

        return op, completions.usage
