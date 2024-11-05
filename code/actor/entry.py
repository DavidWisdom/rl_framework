import logging
import os
import random
import signal
import sys
import psutil

from code.actor import sample_manager
from code.actor.actor import Actor

AGENT_NUM = 2

def select_mempool(actor_id, actor_num, mempool_list):
    mempool_num = len(mempool_list)
    if actor_num % mempool_num and actor_id // mempool_num == actor_num // mempool_num:
        idx = random.randint(0, mempool_num - 1)
    else:
        idx = actor_id % mempool_num
    return mempool_list[idx]

def auto_bind_cpu(actor_id, actor_num):
    p = psutil.Process(os.getpid())
    cpu_ids = p.cpu_affinity() or []
    if actor_id // len(cpu_ids) != actor_num // len(cpu_ids):
        cpu_id = cpu_ids[actor_id % len(cpu_ids)]
        p.cpu_affinity([cpu_id])

def _run(
    actor_id,
    max_frame_num,

    max_episode,
):
    actor = Actor(
        id=actor_id,
        agents=agents,
        max_episode=max_episode,
        is_train=config.IS_TRAIN
    )
    actor.set_sample_manager(sample_manager)
    actor.set_env(env)
    try:
        pass
    except SystemExit:
        raise
    except Exception:
        raise

def run(config):

    pass

def get_config_definition():
    return {
        "id": {
            "value": 0,
            "help": "process idx",
        },
        "mem_pool_addr": {
            "value": "localhost:35200",
            "help": "address of memory pool",
        },
        "model_pool_addr": {
            "value": "localhost:10014",
            "help": "address of model pool",
        },
        "test_config": {
            "value": False,
            "help": "test_config",
        },
        "single_test": {
            "value": False,
            "help": "single test without model pool/mem pool",
        },
        "print_yaml": {"value": False, "help": "print yaml config"},
        "config_path": {
            "value": interface_default_config,
            "help": "config file for interface",
        },
        "gamecore_req_timeout": {
            "value": 30000,
            "help": "millisecond timeout for gamecore to wait reply from server",
        },
        "gc_server_addr": {
            "value": "127.0.0.1:23432",
            "help": "address of gamecore server",
        },
        "aiserver_ip": {"value": "127.0.0.1", "help": "the actor ip"},
        "max_episode": {"value": -1, "help": "max number for run episode"},
        "monitor_server_addr": {
            "value": "127.0.0.1:8086",
            "help": "monitor server addr",
        },
        "runtime_id_prefix": {"value": "actor-1v1", "help": "must not contain '_'"},
        "log_file": {
            "value": "",
            "help": "default(/aiarena/logs/actor/actor_{actor_id}.log)",
        },
        "port_begin": {"value": 35300},
        "max_frame_num": {"value": 20000},
        "debug_log": {"value": False, "help": "use debug log level"},
    }

if __name__ == '__main__':
    def main(_):
        pass
    pass
