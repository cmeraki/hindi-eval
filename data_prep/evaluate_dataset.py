import os
import time
import yaml
import argparse
import numpy as np
from typing import Dict
from datasets import load_from_disk

from .evaluators import GPTEvaluator
from .utils.logger import DataPrepLogger

logger = DataPrepLogger(__name__).get_logger()

def gpt_processor(
        example: Dict,
        message_key: str,
        candidate_1: str,
        candidate_2: str,
        evaluator: GPTEvaluator
) -> Dict:

    winners = []

    for m, c1, c2 in zip(example[message_key], example[candidate_1], example[candidate_2]):
        try:
            decision = evaluator.evaluate(
                original_text=m['content'],
                candidate_1=c1,
                candidate_2=c2
            )

            winner = candidate_1
            if decision == '2' or decision == 2:
                winner = candidate_2
            elif decision == '3' or decision == 3:
                winner = 'both'

            winners.append(winner)
        except Exception as err:
            logger.warn(f'Error: {err}')
            winners.append('')

    example['winners'] = winners
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
        default='./data/translation_eval_results',
        help='Location of config dir',
        type=str,
        required=False
    )
    parser.add_argument(
        '--dataset_path',
        default='./data/translation_eval',
        help='Location of config dir',
        type=str,
        required=False
    )
    args = parser.parse_args()

    logger.info(f'Running with args: {args}')

    evaluator = GPTEvaluator(model_id='gpt-4')
    dataset = load_from_disk(args.dataset_path)

    logger.info(f'Loaded dataset with size: {len(dataset)}')

    with open(os.path.join(args.config_dir, 'evaluate.yml'), 'r') as fp:
        evaluation_config = yaml.safe_load(fp)
        logger.debug(f'Evaluation config: {evaluation_config}')

    for evaluation_task, eval in evaluation_config.items():
        dataset = dataset.select(range(eval['sample_size'])).map(
            lambda x: gpt_processor(
                x,
                message_key=eval['text_key'],
                candidate_1=eval['candidate_1'],
                candidate_2=eval['candidate_2'],
                evaluator=evaluator
            )
        )

        logger.info(f'Saving processed dataset to: {os.path.join(args.save_path, "translate_results")}')

        dataset.save_to_disk(os.path.join(
            args.save_path, f'{evaluation_task}_evaluation_results')
        )
