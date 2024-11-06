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

def run(config):
    config_manager = ConfigControl(config_path)
    config_manager.use_init_model = config.use_init_model
    config_manager.init_model_path = config.init_model_path
    config_manager.load_optimizer_state = config.load_optimizer_state
    config_manager.max_steps = config.max_steps
    config_manager.display_every = config.display_every
    config_manager.save_model_steps = config.save_model_steps
    config_manager.batch_size = config.batch_size
    config_manager.max_sample = config.store_max_sample
    config_manager.send_model_dir = config.backup_model_dir

    try:
        log_level = None
        if config.test_config or config.single_test:
            log_level = "INFO"
        _run(config, config_manager, config.single_test)
    except:
        raise

def get_config_definition():
    return {
        "single_test": {"value": False, "help": "run trainer with random dataset"},
        "print_yaml": {"value": False, "help": "print yaml config"},
        # test_config: 设置batch_size/store_max_sample/display_every小数值以便测试
        "test_config": {"value": False, "help": "modify args for test"},
        "slow_time": {"value": 0.0, "help": "slow time for static policy"},
        "use_init_model": {"value": False},
        "display_every": {"value": 200},
        "save_model_steps": {"value": 1000},
        "max_steps": {"value": 100000000},
        "batch_size": {"value": 512},
        "store_max_sample": {"value": 5000},
        "init_model_path": {"value": "/aiarena/code/learner/model/init/"},
        "load_optimizer_state": {"value": False},
        "use_influxdb_logger": {"value": False},
        "influxdb_ip": {"value": "127.0.0.1"},
        "influxdb_port": {"value": 8086},
        "backup_model_dir": {"value": "/aiarena/backup/"},
    }
