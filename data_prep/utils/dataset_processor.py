import json
import numpy as np
from typing import Dict, List
from datasets import DatasetDict, Dataset

from .logger import DataPrepLogger
from ..translators import BaseTranslator

logger = DataPrepLogger.get_logger(__name__)

with open('../../data/retrieval/cleaned_dataset/dataset.jsonl', 'r') as fp:
    x = fp.read()
    d = []

    for ln in x.split('\n'):
        if not ln:
            continue
        d.append(json.loads(ln))

    retrieval_base = Dataset().from_list(d)


def multi_turn_conv_processor(example: DatasetDict, message_key: str) -> DatasetDict:
    to_translate = [msg['content'] for msg in example[message_key]]
    example['flattened_messages'] = to_translate
    return example


def mmlu_processor(example: DatasetDict, column_names: List[str], translator: BaseTranslator, translator_name: str, batch_size: int=8) -> DatasetDict:
    to_translate = [example[col] for col in column_names]

    translations = translator.translate(to_translate, batch_size=batch_size)

    for elem, col in zip(translations, column_names):
        example[f'translated_response_{col}_{translator_name}'] = elem

    return example


def translator_processor(
        example: Dict,
        message_key: str,
        translator: BaseTranslator,
        translator_name: str
) -> Dict:
    example[f'translated_reponse_{translator_name}'] = translator.translate(
        example[message_key])
    return example


def get_synthetic_data_sys_prompt(base_prompt: str, reference_dataset: Dict, required_format: str) -> Dict:
    subject = np.random.choice(
        list(reference_dataset.keys())
    )
    grade = np.random.choice(
        list(reference_dataset[subject].keys())
    )
    topic = np.random.choice(
        reference_dataset[subject][grade]
    )

    logger.debug(f'Metadata for the prompt: {subject}, {grade}, {topic}')

    sys_prompt = {
        'role': 'system',
        'content': base_prompt.format(
            language='either Devnagri Hindi or Romanized Hindi',
            subject=subject,
            grade=grade,
            topic=topic,
            required_format=required_format
        )
    }

    return [sys_prompt], {'SUBJECT': subject, 'GRADE': grade, 'TOPIC': topic}


def get_retreival_data_sys_prompt(base_prompt: str, reference_dataset: Dict, required_format: str) -> Dict:
    sys_prompt = {
        'role': 'system',
        'content': base_prompt.format(
            num_ques=5,
            language='Devnagri Hindi or Romanized Hindi',
            required_format=required_format
        )
    }

    usr_prompt = {
        'role': 'user',
        'content': retrieval_base.shuffle().select(range(1))['content']
    }

    return [sys_prompt, usr_prompt], {'PASSAGE': usr_prompt['content']}