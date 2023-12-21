import os
import torch
from abc import ABC, abstractmethod
from typing import List
from openai import OpenAI
from torch.cuda import empty_cache
from transformers import AutoProcessor, SeamlessM4TForTextToText
import google.generativeai as genai

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
            torch_dtype=torch.bfloat16
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

        translated_text = self.processor.batch_decode(
            output_tokens, skip_special_tokens=True
        )

        empty_cache()

        return translated_text


class GPTTranslator(BaseTranslator):
    def __init__(self, model_id) -> None:
        from dotenv import load_dotenv
        load_dotenv()

        self.client = OpenAI()
        self.model_id = model_id

    def translate(self, source_text: List[str]) -> List[str]:
        system_prompt = {
            "role": "system",
            "content": "You are an expert tranlator who traslates given text in English to colloquial Devnagri Hindi. You output nothing except the translation."
        }

        total_tokens = 0
        translated_text = []

        for txt in source_text:
            user_prompt = {
                'role': 'user',
                'content': f'Translate the following to Hindi: \n{txt}'
            }
            completions = self.client.chat.completions.create(
                model=self.model_id,
                messages=[system_prompt, user_prompt]
            )
            total_tokens += completions.usage.total_tokens
            translated_text.append(completions.choices[0].message.content)

        logger.debug(f'Used {total_tokens} tokens for {self.model_id}')

        return translated_text


class GeminiTranslator(BaseTranslator):
    def __init__(self, model_id: str) -> None:
        from dotenv import load_dotenv

        load_dotenv()
        GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
        genai.configure(api_key=GOOGLE_API_KEY)

        self.model = genai.GenerativeModel(model_id)

    def translate(self, source_text: List[str]) -> str:
        chat = self.model.start_chat(
            history=[]
        )

        translated_text = []

        for txt in source_text:
            try:
                user_prompt = {
                    'role': 'user',
                    'parts': [f'Translate the following to Hindi: \n{txt}']
                }
                logger.debug(f'Prompt sent to gemini: {user_prompt}')
                response = chat.send_message(user_prompt)

                translated_text.append(response.text)
            except Exception as err:
                logger.warn(f'Error in {txt} with {err}')
                translated_text.append('')

        return translated_text
