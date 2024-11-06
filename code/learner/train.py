import os
from code.common.algorithm import Algorithm
from code.common.config import Config
from code.common.benchmark import Benchmark
from code.common.config_control import ConfigControl
from code.common.datasets import NetworkDatasetRandom, NetworkDatasetZMQ
from code.learner.model_manager import ModelManager
from code.common.node_info_ddp import NodeInfo
from code.common.offline_rlinfo_adapter import OfflineRlInfoAdapter

config_path = os.path.join(os.path.dirname(__file__), "config", "common.conf")

def _run(model_config, framework_config, single_test):
    """
        model_config: 模型配置
        framework_config: 框架配置, 为框架的ConfigControl
        single_test: 单独测试learner
        """

    config_manager = framework_config  # alias
    os.makedirs(config_manager.save_model_dir, exist_ok=True)
    os.makedirs(config_manager.train_dir, exist_ok=True)
    os.makedirs(config_manager.send_model_dir, exist_ok=True)

    if single_test:
        config_manager.push_to_modelpool = False
        config_manager.distributed_backend = "none"

    if model_config.test_config:
        config_manager.distributed_backend = "none"
        config_manager.batch_size = 2
        config_manager.display_every = 1
        config_manager.max_sample = 10
        config_manager.max_steps = 5

    code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    model_manager = ModelManager(
        code_dir,
        os.path.join("code", "actor", "model", "init"),
        config_manager.push_to_modelpool,
        save_checkpoint_dir=config_manager.save_model_dir,
        backup_checkpoint_dir=config_manager.send_model_dir,
        load_optimizer_state=config_manager.load_optimizer_state,
    )

    adapter = OfflineRlInfoAdapter(Config.data_shapes)
    node_info = NodeInfo()

    if single_test:
        dataset = NetworkDatasetRandom(config_manager, adapter)
    else:
        dataset = NetworkDatasetZMQ(
            config_manager, adapter, port=config_manager.ports[node_info.local_rank]
        )

    benchmark = Benchmark(
        Algorithm(),
        dataset,
        model_manager,
        config_manager,
        node_info,
        slow_time=model_config.slow_time,
    )
    benchmark.run()

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
