import logging
import os
import random
import signal
import sys
import psutil

from code.actor import sample_manager
from code.actor.actor import Actor
from code.actor.custom import Agent
from code.actor.model import Model
from env.tic_tac_toe import TicTacToe

AGENT_NUM = 2
work_dir = os.path.dirname(os.path.abspath(__file__))

# sys.path.append("/aiarena/code/") add common to path
sys.path.append(os.path.dirname(work_dir))

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
    config_path,
    model_pool_addr,
    single_test,
    port_begin,
    gc_server_addr,
    gamecore_req_timeout,
    max_frame_num,
    runtime_id_prefix,
    aiserver_ip,
    mem_pool_addr_list,
    max_episode,
    monitor_server_addr,
    config,
):
    if config.auto_bind_cpu:
        auto_bind_cpu(actor_id, config.actor_num)

    # chdir to work_dir to access the config.json with relative path
    os.chdir(work_dir)

    agents = []
    game_id_init = "None"
    main_agent = random.randint(0, AGENT_NUM - 1)

    for i in range(AGENT_NUM):
        agents.append(
            Agent(
                Model(),
                model_pool_addr.split(";"),
                config=config,
                keep_latest=(i == main_agent),
                single_test=single_test,
            )
        )

    addrs = []
    for i in range(AGENT_NUM):
        addrs.append(f"tcp://0.0.0.0:{port_begin + actor_id * AGENT_NUM + i}")

    runtime_id = f"{runtime_id_prefix.replace('_', '-')}-{actor_id}"

    env = TicTacToe()

    mempool = select_mempool(actor_id, config.actor_num, mem_pool_addr_list)
    sample_manager = SampleManager(
        mem_pool_addr=mempool,
        mem_pool_type="zmq",
        num_agents=AGENT_NUM,
        game_id=game_id_init,
        single_test=single_test,
        data_shapes=config.data_shapes,
        lstm_time_steps=config.LSTM_TIME_STEPS,
        gamma=config.GAMMA,
        lamda=config.LAMDA,
    )

    actor = Actor(
        actor_id=actor_id,
        agents=agents,
        max_episode=max_episode,
        is_train=config.IS_TRAIN,
    )
    actor.set_sample_manager(sample_manager)
    actor.set_env(env)
    actor.run(eval_freq=config.EVAL_FREQ)

def run(config):
    mem_pool_addr_list = config.mem_pool_addr.strip().split(";")
    monitor_ip = mem_pool_addr_list[0].split(":")[0]
    monitor_server_addr = f"{monitor_ip}:8086"
    from code.common.config import Config
    try:
        log_file = config.log_file or "/aiarena/logs/actor/actor_{}.log".format(
            config.id
        )
        log_level = None
        if config.debug_log:
            log_level = "DEBUG"
        elif config.single_test:
            log_level = "INFO"
        elif config.test_config:
            log_level = "INFO"
            config.max_frame_num = 1000

        _run(
            config.id,
            config.config_path,
            config.model_pool_addr,
            config.single_test,
            config.port_begin,
            config.gc_server_addr,
            config.gamecore_req_timeout,
            config.max_frame_num,
            config.runtime_id_prefix,
            config.aiserver_ip,
            mem_pool_addr_list,
            config.max_episode,
            monitor_server_addr,
            Config,
        )
    except SystemExit:
        raise
    except Exception:
        raise

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
        "aiserver_ip": {"value": "127.0.0.1", "help": "the actor ip"},
        "max_episode": {"value": -1, "help": "max number for run episode"},
        "runtime_id_prefix": {"value": "actor-1v1", "help": "must not contain '_'"},
        "log_file": {
            "value": "",
            "help": "default(/aiarena/logs/actor/actor_{actor_id}.log)",
        },
        "port_begin": {"value": 35300},
        "max_frame_num": {"value": 20000},
        "debug_log": {"value": False, "help": "use debug log level"},
    }