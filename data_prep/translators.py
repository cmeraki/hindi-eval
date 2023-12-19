from abc import ABC, abstractmethod
from typing import List
from openai import OpenAI
from torch.cuda import empty_cache
from transformers import AutoProcessor, SeamlessM4TForTextToText

from .logger import DataPrepLogger

logger = DataPrepLogger(__name__).get_logger()

class BaseTranslator(ABC):
    @abstractmethod
    def translate(self):
        pass


class SeamlessM4TTranslator(BaseTranslator):
    def __init__(self, model_id: str) -> None:
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            src_lang='eng'
        )
        self.model = SeamlessM4TForTextToText.from_pretrained(
            model_id,
            device_map='cuda:0',
            max_new_tokens=4096,
        )

    def translate(self, source_text: List[str]) -> List[str]:
        logger.debug(f'Passing {len(source_text)} sequences as input batch')

        input_tokens = self.processor(
            text=source_text,
            return_tensors='pt'
        )

        output_tokens = self.model.generate(
            **input_tokens.to('cuda:0'),
            tgt_lang='hin'
        )

        output_text = self.processor.batch_decode(
            output_tokens, skip_special_tokens=True
        )

        empty_cache()

        return output_text


class GPTTranslator(BaseTranslator):
    def __init__(self, model_id) -> None:
        from dotenv import load_dotenv
        load_dotenv()

        self.client = OpenAI()
        self.model_id = model_id

    def translate(self, source_text: List[str]) -> str:
        completions = self.client.chat.completions.create(
            model=self.model_id,
            messages=source_text
        )

        logger.debug(f'Used {completions.usage.total_tokens} tokens for {self.model_id}')

        return completions.choices[0].message.content
