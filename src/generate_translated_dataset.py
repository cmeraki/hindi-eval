"""
Translates dataset using LLMs

Relevant configs
1. `.configs.translators.translator_engines` -> Configuration for which translators to use
2. `.configs.dataset.translation_datasets` -> Configuration for which dataset to translate
"""

import os
import gc
import argparse
from datetime import datetime
from torch.cuda import empty_cache
from datasets import load_dataset, get_dataset_config_names
from functools import partial

from .utils.logger import DataPrepLogger
from .configs.translators import translator_engines
from .configs.dataset import translation_datasets

logger = DataPrepLogger(__name__).get_logger()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_path',
        default='./data/processed/translations',
        help='Location of path to save the generated dataset',
        type=str,
        required=False
    )
    args = parser.parse_args()

    logger.info(f'Running with args: {args}')

    translators = [(t, c) for t, c in translator_engines.items() if c.enabled]
    processing_datasets = [(d, c) for d, c in translation_datasets.items() if c.enabled]

    logger.info(f'Translating {[d[0] for d in processing_datasets]} datasets using {[t[0] for t in translators]} models')
    run_time = datetime.strftime(datetime.now(), '%Y%m%d-%H%M')

    for translator_name, translator_config in translators:
        logger.info(f'Starting translation on {translator_name}')

        # initialize the engine
        translator_config.engine(**translator_config.kwargs)

        # Iterate over all datasets enabled
        for dataset_name, dataset_config in processing_datasets:

            confs = get_dataset_config_names(dataset_config.dataset_id)
            if '*' not in dataset_config.config:
                confs_not_found = [c for c in dataset_config.config if c not in confs]
                assert len(confs_not_found) == 0, f'{len(confs_not_found)} configs ({confs_not_found}) not found in dataset config on hugginface'
                confs = dataset_config.config

            logger.info(f'{len(confs)} configs of the dataset will be loaded')

            # Iterate over all configs mentioned
            for conf in confs:
                logger.info(f'Loading the dataset: {dataset_name} with config {conf}')
                func = partial(load_dataset)

                if dataset_config.split and dataset_config.split != '*':
                    func = partial(func, split=dataset_config.split)

                message_key = dataset_config.text_key

                ds = func(dataset_config.dataset_id, conf)

                if dataset_config.sample_size != -1:
                    sample_size = min(dataset_config.sample_size, ds.num_rows)
                    logger.info(f'Selecting {sample_size} rows from the dataset')
                    ds = ds.shuffle().select(range(sample_size))

                for preprocess_func in dataset_config.preprocess_func:
                    logger.info(f'Applying preprocessing functions to the dataset')
                    ds = ds.map(
                        lambda x: preprocess_func(x, message_key)
                    )

                ds = ds.map(
                    lambda x: dataset_config.transform_func(
                        x, translator=translator_config.engine, translator_name=translator_name
                    )
                )

                logger.info('Saving the dataset')

                ds.save_to_disk(
                    os.path.join(
                        args.save_path,
                        run_time,
                        f'{translator_name}_{dataset_name}',
                        conf if conf != '*' else 'all',
                    )
                )

        del (translator_config.engine)
        empty_cache()
        gc.collect()
