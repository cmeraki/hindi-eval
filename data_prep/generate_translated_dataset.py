import os
import yaml
import argparse
from typing import Dict
from datetime import datetime
from datasets import load_dataset

from .translators import SeamlessM4TTranslator, GPTTranslator, GeminiTranslator, BaseTranslator
from .logger import DataPrepLogger

logger = DataPrepLogger(__name__).get_logger()

def multi_turn_conv_processor(example: Dict, message_key: str) -> Dict:
    to_translate = [msg['content'] for msg in example[message_key]]
    example['flattened_messages'] = to_translate
    return example

def translator_processor(
        example: Dict,
        message_key: str,
        translator: BaseTranslator,
        translator_name: str
) -> Dict:
    example[f'translated_reponse_{translator_name}'] = translator.translate(example[message_key])
    return example

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_dir',
        help='Location of config dir',
        type=str,
        required=True
    )
    parser.add_argument(
        '--save_path',
        default='./data/',
        help='Location of config dir',
        type=str,
        required=False
    )
    args = parser.parse_args()

    logger.info(f'Running with args: {args}')

    with open(os.path.join(args.config_dir, 'dataset.yml'), 'r') as fp:
        dataset_config = yaml.safe_load(fp)

    t1 = SeamlessM4TTranslator(model_id='facebook/hf-seamless-m4t-large')
    t2 = GPTTranslator(model_id='gpt-3.5-turbo-1106')
    t3 = GeminiTranslator(model_id='gemini-pro')

    # Iterate over all datasets present
    logger.info(f'Datasets mentioned in config: {len(dataset_config.keys())}')

    for dataset_name, dc in dataset_config.items():
        if not dc['process']:
            logger.info(f'Skipping {dataset_name} dataset')
            continue

        if 'config' in dc.keys():
            dataset = load_dataset(dataset_name, dc['config'])
        else:
            dataset = load_dataset(dataset_name)

        message_key = dc['text_key']
        dataset = dataset[dc['split']].shuffle().select(range(int(dc['sample_size'])))
        dataset = dataset.map(
            lambda x: multi_turn_conv_processor(x, message_key),
            num_proc=20
        )

        logger.info(f'Starting processing for seamless m4t')
        dataset = dataset.map(
            lambda x: translator_processor(x, 'flattened_messages', t1, 'm4t')
        )

        logger.info(f'Starting processing for GPT API')
        dataset = dataset.map(
            lambda x: translator_processor(x, 'flattened_messages', t2, 'gpt')
        )

        logger.info(f'Starting processing for Gemini API')
        dataset = dataset.map(
            lambda x: translator_processor(x, 'flattened_messages', t3, 'gemini')
        )

        logger.info(os.path.join(
            args.save_path,
            'translation_eval',
            datetime.strftime(datetime.now(), '%Y%m%d%H'),
        ))

        dataset.save_to_disk(
            os.path.join(
                args.save_path,
                'translation_eval',
                datetime.strftime(datetime.now(), '%Y%m%d%H'),
            )
        )
