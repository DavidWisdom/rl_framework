import sys
import time
import os

from multiprocessing import Process
from code.actor.entry import run as actor_run
from code.actor.entry import get_config_definition as actor_config_definition
from code.learner.train import run as learner_run
from code.learner.train import get_config_definition as learner_config_definition
from process import ModelPoolProcess

script_dir = os.path.dirname(os.path.abspath(__file__))
config_definition = {
    "print_yaml": {"value": False, "help": "print yaml config"},
    "test_config": {"value": False, "help": "test mode for quick test"},
    "actor_only": {
        "value": False,
    },
    "learner_only": {
        "value": False,
    },
    "influxdb_port": {
        "value": 8086,
    },
    "input_learner_list": {
        "value": os.path.join(script_dir, "learner.iplist"),
        "help": "input learner list from platform",
        "env_alias": ["input_learner_list"],
    },
    "file_save_path": {
        "value": "/mnt/ramdisk/model",
        "env_alias": ["MODEL_POOL_FILE_SAVE_PATH"],
    },
    "actor_num": {
        "value": 1,
        "env_alias": ["CPU_NUM", "ACTOR_NUM"],
    },
    "run_timeout": {
        "value": 300,
    },
}

def test_actor(config):
    config.single_test = True
    config.max_episode = 1
    config.max_frame_num = 1000
    process = Process(target=actor_run, args=(config,))
    process.daemon = True
    process.run()

def test_learner(config):
    pass

def start_actor(config):
    pass

def start_learner(config):
    pass

def overwrite_config(config):
    if config.test_config:
        config.actor.test_config = True
        config.learner.test_config = True

def start_model_pool(config, master_ip):
    model_pool = Process(
        target=run_process, args=(ModelPoolProcess, "gpu", master_ip), name="model_pool"
    )
    model_pool.start()
    return [model_pool]

if __name__ == '__main__':
    def main(_):
        pass

