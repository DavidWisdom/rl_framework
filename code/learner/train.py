import os
import sys
import logging

from code.common.algorithm import Algorithm
from code.learner.benchmark import Benchmark
from code.learner.datasets import NetworkDatasetRandom


def _run(model_config, framework_config, single_test):
    config_manager = framework_config
    os.makedirs(config_manager.save_model_dir, exist_ok=True)
    os.makedirs(config_manager.train_dir, exist_ok=True)
    os.makedirs(config_manager.send_model_dir, exist_ok=True)

    if single_test:
        dataset = NetworkDatasetRandom(config_manager, adapter)
    else:
        dataset = None
    benchmark = Benchmark(
        Algorithm(),
        dataset,
        model_manager,
        config_manager,

    )
    pass

def get_config_definition():
    return {
        "single_test": {"value": False, "help": "run trainer with random dataset"},
        "use_init_model": {"value": False},
        "store_max_sample": {"value": False},
    }

if __name__ == '__main__':
    def main(_):
        pass
    pass
