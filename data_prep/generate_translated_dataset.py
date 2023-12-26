import os
import argparse
from datetime import datetime
from datasets import load_dataset
from functools import partial

from .utils.logger import DataPrepLogger
from .configs.translators import translator_engines
from .configs.dataset import translation_datasets

logger = DataPrepLogger(__name__).get_logger()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_path',
        default='./data/',
        help='Location of config dir',
        type=str,
        required=False
    )
    args = parser.parse_args()

    logger.info(f'Running with args: {args}')

    translators = [(t, c) for t, c in translator_engines.items() if c.enabled]
    processing_datasets = [(d, c) for d, c in translation_datasets.items() if c.enabled]

    logger.info(f'Translating {[d[0] for d in processing_datasets]} datasets using {[t[0] for t in translators]} models')

    # Iterate over all datasets enabled
    for dataset_name, dataset_config in processing_datasets:
        logger.info(f'Loading the dataset: {dataset_name}')
        func = partial(load_dataset)

        if dataset_config.config and dataset_config.config != '*':
            func = partial(
                func,
                name=dataset_config.config
            )
        if dataset_config.split and dataset_config.split != '*':
            func = partial(
                func,
                split=dataset_config.split
            )

        message_key = dataset_config.text_key

        ds = func(dataset_config.dataset_id)
        if dataset_config.sample_size != -1:
            sample_size = min(dataset_config.sample_size, ds.num_rows)
            logger.info(f'Selecting {sample_size} rows from the dataset')
            ds = ds.shuffle().select(range(sample_size))

        for preprocess_func in dataset_config.preprocess_func:
            logger.info(f'Applying preprocessing functions to the dataset')
            ds = ds.map(
                lambda x: preprocess_func(x, message_key)
            )

        for translator_name, translator_config in translators:
            logger.info(f'Starting translation on {translator_name}')
            translator_config.engine() # initialize the engine

            ds = ds.map(
                lambda x: dataset_config.transform_func(
                    x, translator=translator_config.engine, translator_name=translator_name
                )
            )

        logger.info(os.path.join(
            args.save_path,
            'translation_eval',
            datetime.strftime(datetime.now(), '%Y%m%d-%H%M'),
        ))

        ds.save_to_disk(
            os.path.join(
                args.save_path,
                'translation_eval',
                dataset_name,
                datetime.strftime(datetime.now(), '%Y%m%d%H'),
            )
        )
