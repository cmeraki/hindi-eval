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
    config: List[str]
    split: str = None
    text_key: str = None
    preprocess_func: List = []
    transform_func: object = []

translation_datasets = {
    'no_robots': TranslationDatasets(
        dataset_id='HuggingFaceH4/no_robots',
        sample_size=200,
        enabled=False,
        config=['*'],
        split='train_sft',
        preprocess_func=[multi_turn_conv_processor],
        transform_func=partial(translator_processor, message_key='flattened_messages')
    ),
    'wikipedia': TranslationDatasets(
        dataset_id='wikimedia/wikipedia',
        sample_size=100,
        enabled=False,
        config=['20231101.en'],
        split='train',
    ),
    'mmlu_train': TranslationDatasets(
        dataset_id='lukaemon/mmlu',
        sample_size=-1,
        enabled=True,
        config=['*'],
        split='test',
        transform_func=partial(mmlu_processor, column_names=['input', 'A', 'B', 'C', 'D'], batch_size=2)
    ),
}
