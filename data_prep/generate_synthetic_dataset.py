"""
Generator of synthetic data

Uses GPT4 to generate dataset
"""

import os
import json
import openai
import backoff
from typing import Tuple, Dict, List
from datetime import datetime
from argparse import ArgumentParser

from .utils.logger import DataPrepLogger
from .configs.synthetic_dataset import GenerationConfiguration, synthetic_dataset_models

logger = DataPrepLogger(__name__).get_logger()


def synth_save_to_disk(base_path: str, generated_dataset: List):
    """
    Saves data to disk in a sequential file by removing elements from the list
    The function will flush the contents of the List to the disk

    Args:
        base_path (str): Location of the base path where the data should be written
        generated_dataset (List): Each element should be a valid Dict
    """

    os.makedirs(base_path, exist_ok=True)

    logger.info(f'Number of rows: {len(generated_dataset)}')

    with open(os.path.join(base_path, 'dataset.jsonl'), 'a', encoding='utf-8') as fp:
        while generated_dataset:
            fp.write(json.dumps(generated_dataset.pop(), ensure_ascii=False))
            fp.write('\n')

class GPTGenerator():
    def __init__(self, model_id) -> None:
        from dotenv import load_dotenv
        load_dotenv()

        self.client = openai.OpenAI()
        self.model_id = model_id

    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=300)
    def __call__(self, messages: List[str], temperature: float = 1.4) -> Tuple[Dict, any]:

        completions = self.client.chat.completions.create(
            model=self.model_id,
            response_format={'type': 'json_object'},
            messages=messages,
            temperature=temperature,
            max_tokens=2048
        )

        if completions.choices[0].finish_reason == 'length':
            raise IOError(f'Reached maximum output length, output format is not reliable. {completions.choices[0].message.content}')

        op = json.loads(completions.choices[0].message.content)

        logger.debug(f'Prompts: {messages}, output: {op}')
        logger.debug(f'Tokens used in generation using {self.model_id}: {completions.usage}')

        return op, completions.usage


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--save_path',
        default='./data/synthetic_data/',
        help='Location of config dir',
        type=str,
        required=False
    )
    args = parser.parse_args()

    logger.info(f'Running with args: {args}')

    generator = GPTGenerator(model_id=GenerationConfiguration.model_id)
    run_time = datetime.strftime(datetime.now(), '%Y%m%d-%H%M')

    for synth_ds_name, synth_ds in synthetic_dataset_models.items():
        total_usage = {
            'input': 0,
            'output': 0,
        }

        if not synth_ds.enabled:
            logger.info(f'Skipping {synth_ds_name}')
            continue

        generated_dataset = []
        logger.info(f'Generating synthetic dataset for {synth_ds_name}')

        idx = 0
        for messages, metadata in synth_ds.preprocess_func(synth_ds.system_prompt, synth_ds.reference_dataset, synth_ds.required_format, synth_ds.sample_size):
            try:
                datapoint, usage = generator(
                    messages=messages,
                    temperature=GenerationConfiguration.temperature
                )
                total_usage['input'] += usage.prompt_tokens
                total_usage['output'] += usage.completion_tokens

                if synth_ds.output_type.value == 'multi': # if a single prompt is used to generate multiple questions, datapoint will have 1 key with a list as value
                    logger.debug(f'Processing multi point dataset')
                    assert synth_ds.response_model.model_validate(datapoint), "Response by the model is not in the valid schema"

                    for _, v in datapoint.items():
                        for d in v:
                            d.update({**metadata})
                            generated_dataset.append(d)

                else: # datapoint in itself has a single question/answer
                    assert synth_ds.response_model.model_validate(datapoint), "Response by the model is not in the valid schema"
                    datapoint.update({
                        **metadata
                    })
                    generated_dataset.append(datapoint)

                logger.debug(f'Used cumulative tokens: {total_usage}')

            except openai.RateLimitError as err:
                logger.error(f'Reached rate limit: {err}')
                break
            except Exception as err:
                logger.error(f'Raised error: {err}')

            finally:
                if idx % 10 == 0:
                    synth_save_to_disk(
                        base_path=os.path.join(
                            args.save_path, run_time, f'{synth_ds_name}'),
                        generated_dataset=generated_dataset
                    )
                idx += 1

        synth_save_to_disk(
            base_path=os.path.join(args.save_path, run_time, f'{synth_ds_name}'),
            generated_dataset=generated_dataset
        )
