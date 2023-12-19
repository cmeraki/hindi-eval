import os
import yaml
import argparse
from tqdm import tqdm
from typing import Dict
from datasets import load_dataset

from .utils import get_prompt
from .translators import SeamlessM4TTranslator, GPTTranslator, BaseTranslator
from .logger import DataPrepLogger

logger = DataPrepLogger(__name__).get_logger()


def m4t_processor(example: Dict, message_key: str, translator: BaseTranslator) -> Dict:
    to_translate = []
    for msg in example[message_key]:
        to_translate.append(msg['content'])

    example['translated_reponse_m4t'] = translator.translate(to_translate)
    return example

def gpt_processor(example: Dict, message_key: str, translator: BaseTranslator) -> Dict:
    translate_prompts = get_prompt(example, message_key)

    translated_reponse = []
    for prompt in translate_prompts:
        logger.debug(f'Prompt created to send to GPT: {prompt} for {example["prompt_id"]}')
        translated_reponse.append(
            translator.translate(prompt)
        )

    example['translated_reponse_gpt'] = translated_reponse
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

    t1 = SeamlessM4TTranslator(model_id='facebook/hf-seamless-m4t-medium')
    t2 = GPTTranslator(model_id='gpt-3.5-turbo-1106')

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

        dataset = dataset[dc['split']].shuffle().select(range(int(dc['sample_size'])))
        message_key = dc['text_key']

        logger.info(f'Starting processing for seamless m4t')
        dataset = dataset.map(
            lambda x: m4t_processor(x, message_key, t1)
        )

        logger.info(f'Starting processing for GPT APIs')
        dataset = dataset.map(
            lambda x: gpt_processor(x, message_key, t2)
        )

        logger.info(f'Saving processed dataset to: {os.path.join(args.save_path, "translate_results")}')

        dataset.save_to_disk(os.path.join(args.save_path, 'translate_results'))
