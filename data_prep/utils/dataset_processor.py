from typing import Dict, List
from datasets import DatasetDict

from ..translators import BaseTranslator

def multi_turn_conv_processor(example: DatasetDict, message_key: str) -> DatasetDict:
    to_translate = [msg['content'] for msg in example[message_key]]
    example['flattened_messages'] = to_translate
    return example


def mmlu_processor(example: DatasetDict, column_names: List[str], translator: BaseTranslator, translator_name: str) -> DatasetDict:
    to_translate = [example[col] for col in column_names]

    translations = translator.translate(to_translate, batch_size=8)

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
