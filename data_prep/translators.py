import os
import torch
from textwrap import dedent
from abc import ABC, abstractmethod
from typing import List
from openai import OpenAI
from torch.cuda import empty_cache
from transformers import (
    AutoProcessor,
    SeamlessM4TForTextToText,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from optimum.bettertransformer import BetterTransformer
import google.generativeai as genai

from .utils.logger import DataPrepLogger
logger = DataPrepLogger(__name__).get_logger()

class BaseTranslator(ABC):
    @abstractmethod
    def translate(self):
        pass


class SeamlessM4TTranslator(BaseTranslator):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def __call__(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            src_lang='eng'
        )
        self.model = SeamlessM4TForTextToText.from_pretrained(
            self.model_id,
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


class HFTranslator(BaseTranslator):
    def __init__(self, model_id: str, prompt_template: str=None) -> None:
        self.model_id = model_id
        self.prompt_template = prompt_template

    def __call__(self, **model_kwargs):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            fast=True,
            padding='max_length'
        )
        logger.info(f'Recieved model kwargs: {model_kwargs}')
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map='cuda:0',
            **model_kwargs
        )

        try:
            logger.info(f'Trying to convert the model to bettertransformer')
            model = BetterTransformer.transform(model, keep_original_model=False)

        except Exception as err:
            logger.warn(f'Not able to convert the model to transformer. Skipping')

        self.prompt_template = self.prompt_template
        if not self.prompt_template:
            self.prompt_template = dedent("""
            [INST] <<SYS>>
            You are an expert tranlator who traslates given text in English to colloquial Devnagri Hindi. You output nothing except the translation.
            <</SYS>>
            [/INST]
            <|user|> Translate "{prompt}" into Devnagri Hindi
            <|assistant|> Here is the translation in Hindi:
            """)

        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=2048,
            repetition_penalty=1.1
        )

        self.pipe.tokenizer.padding_side = 'left'
        self.pipe.tokenizer.pad_token_id = model.config.eos_token_id

    def translate(self, source_text: List[str], batch_size: int=4) -> List[str]:
        to_translate = [f'{self.prompt_template.format(prompt=t)}' for t in source_text]

        logger.info(f'First prompt sent to the model: {to_translate[0]}')
        outputs = self.pipe(
            to_translate,
            batch_size=batch_size
        )

        empty_cache()

        return [o[0]['generated_text'][len(i):] for i, o in zip(to_translate, outputs)]
