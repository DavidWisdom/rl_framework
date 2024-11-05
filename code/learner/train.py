import os
import sys
import logging

from code.common.algorithm import Algorithm
from code.learner.benchmark import Benchmark

def _run(model_config, framework_config, single_test):
    benchmark = Benchmark(
        Algorithm(),

    )
    pass

def get_config_definition():
    return {
        "single_test": {"value": False, "help": "run trainer with random dataset"},
        "use_init_model": {"value": False},
        "store_max_sample": {"value": False},
    }

if __name__ == '__main__':
    pass
