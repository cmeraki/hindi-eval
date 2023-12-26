"""
Holds the config for all the datasets that needs to be translated
"""

from typing import List
from functools import partial
from pydantic import BaseModel

from ..utils.dataset_processor import mmlu_processor, multi_turn_conv_processor, translator_processor

class TranslationDatasets(BaseModel):
    dataset_id: str
    sample_size: int
    enabled: bool
    split: str = None
    text_key: str = None
    config: str = None
    preprocess_func: List = []
    transform_func: object = []

translation_datasets = {
    'no_robots': TranslationDatasets(
        dataset_id='HuggingFaceH4/no_robots',
        split='train_sft',
        sample_size=200,
        enabled=False,
        preprocess_func=[multi_turn_conv_processor],
        transform_func=partial(translator_processor, message_key='flattened_messages')
    ),
    'wikipedia': TranslationDatasets(
        dataset_id='wikimedia/wikipedia',
        config='20231101.en',
        split='train',
        enabled=False,
        sample_size=100,
    ),
    'mmlu_train_high_school_european_history': TranslationDatasets(
        dataset_id='lukaemon/mmlu',
        config='high_school_european_history',
        enabled=True,
        split='train',
        sample_size=100,
        transform_func=partial(mmlu_processor, column_names=['input', 'A', 'B', 'C', 'D'], batch_size=1)
    ),
    'mmlu_train_high_school_microeconomics': TranslationDatasets(
        dataset_id='lukaemon/mmlu',
        config='high_school_microeconomics',
        enabled=True,
        split='train',
        sample_size=100,
        transform_func=partial(mmlu_processor, column_names=['input', 'A', 'B', 'C', 'D'], batch_size=1)
    ),
}
