"""
Generator of synthetic data

Uses GPT4 to generate dataset
"""

import os
import json
import openai
import backoff
import numpy as np
from tqdm import tqdm
from typing import Tuple, Dict
from datetime import datetime
from datasets import Dataset
from argparse import ArgumentParser

from .utils.logger import DataPrepLogger
from .configs.synthetic_dataset import GenerationConfiguration, synthetic_dataset_models

logger = DataPrepLogger(__name__).get_logger()


def get_sys_prompt(base_prompt: str, reference_dataset: Dict, required_format: str) -> Dict:
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

    return sys_prompt, {'SUBJECT': subject, 'GRADE': grade, 'TOPIC': topic}


class GPTGenerator():
    def __init__(self, model_id) -> None:
        from dotenv import load_dotenv
        load_dotenv()

        self.client = openai.OpenAI()
        self.model_id = model_id

    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=300)
    def __call__(self, system_prompt: str, user_prompt: str = None, temperature: float = 1.4) -> Tuple[Dict, any]:
        messages = [system_prompt]
        if user_prompt:
            messages.append(user_prompt)

        completions = self.client.chat.completions.create(
            model=self.model_id,
            response_format={'type': 'json_object'},
            messages=messages,
            temperature=temperature
        )

        if completions.choices[0].finish_reason == 'length':
            raise IOError(f'Reached maximum output length, output format is not reliable. {completions.choices[0].message.content}')

        op = json.loads(completions.choices[0].message.content)

        logger.debug(f'Tokens used in generation using {self.model_id}: {completions.usage}')
        # logger.debug(f'Completion output from Open AI APIs: {completions}')

        return op, completions.usage


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--save_path',
        default='./synthetic_data/',
        help='Location of config dir',
        type=str,
        required=False
    )
    args = parser.parse_args()

    logger.info(f'Running with args: {args}')

    generator = GPTGenerator(model_id=GenerationConfiguration.model_id)

    total_usage = {
        'input': 0,
        'output': 0,
    }
    run_time = datetime.strftime(datetime.now(), '%Y%m%d-%H%M')

    for synth_ds_name, synth_ds in synthetic_dataset_models.items():
        generated_dataset = []
        logger.info(f'Generating synthetic dataset for {synth_ds.name}')

        for idx in tqdm(range(synth_ds.sample_size)):
            sys_prompt, metadata = get_sys_prompt(synth_ds.system_prompt, synth_ds.reference_dataset, synth_ds.required_format)

            try:
                datapoint, usage = generator(
                    system_prompt=sys_prompt,
                    temperature=GenerationConfiguration.temperature
                )
                total_usage['input'] += usage.prompt_tokens
                total_usage['output'] += usage.completion_tokens

                logger.debug(f'Return datapoint: {datapoint}')

                assert synth_ds.response_model.model_validate(datapoint), "Response by the model is not in the valid dataform"

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

        generated_dataset = Dataset.from_list(generated_dataset)
        logger.info(f'Number of rows: {generated_dataset.num_rows}')
        generated_dataset.save_to_disk(
            os.path.join(
                args.save_path,
                run_time,
                f'{synth_ds_name}'
            )
        )
