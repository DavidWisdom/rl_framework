import os
import torch
import numpy as np
import time

from code.common.datasets import NetworkDataset


class Benchmark(object):
    def __init__(self, network, dataset, model_manager, config_manager, node_info = None, slow_time: float = 0.0):
        self.model_manager = model_manager
        self.config_manager = config_manager
        self.slow_time = slow_time

        self.dataset_base = dataset
        self.init_env(node_info)
        self._init_model(network)
        if torch.cuda.is_available():
            pass

    def init_env(self, node_info):
        self.node = node_info
        self.distributed_backend = self.config_manager.distributed_backend
        if torch.cuda.is_available():
            self.device = torch.device('cuda', self.node.local_rank)
            torch.cuda.set_device(self.node.local_rank)
        else:
            self.device = torch.device('cpu', self.node.local_rank)

    def _default_optimizer(self):
        initial_lr = self.net.learning_rate
        params = self.net.parameters()
        optimizer = torch.optim.Adam(params=params, lr=initial_lr, betas=(0.9, 0.999), eps=1e-8)
        return optimizer

    def _init_model(self, network):
        self.local_step = 0
        self.net = network.to(self.device)
        if self.config_manager.channels_last:
            self.net = self.net.to(memory_format=torch.channels_last)
        if self.config_manager.use_compile and hasattr(torch, 'compile'):
            self.net = torch.compile(self.net)
        get_optimizer = getattr(self.net, "get_optimizer", None)
        if callable(get_optimizer):
            self.optimizer = self.net.get_optimizer()
        else:
            self.optimizer = self._default_optimizer()
        self.parameters = [
            p
            for param_group in self.optimizer.param_groups
            for p in param_group["params"]
        ]
        if self.config_manager.use_init_model:
            model_checkpoint_path = os.path.join(
                self.config_manager.init_model_path, "model.pth"
            )
            ckpt_step = self.model_manager.restore_model_and_optimizer(
                self.net, self.optimizer, model_checkpoint_path
            )
            self.local_step = 0 if ckpt_step is None else ckpt_step
        self.init_step = self.local_step
        get_lr_scheduler = getattr(self.net, 'get_lr_scheduler', None)
        if callable(get_lr_scheduler):
            self.lr_scheduler = self.net.get_lr_scheduler(
                self.optimizer, self.local_step
            )
        else:
            self.lr_scheduler = None

        if self.distributed_backend == "horovod" and self.node.has_hvd:
            self.net.to(memory_format=torch.contiguous_format)
            import horovod.torch as hvd

            self.optimizer = hvd.DistributedOptimizer(self.optimizer)
            hvd.broadcast_parameters(self.net.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
            self.net_wrapper = self.net
        elif self.distributed_backend == "ddp":
            from torch.nn.parallel import DistributedDataParallel as DDP

            self.net_wrapper = DDP(
                self.net,
                [
                    self.node.local_rank,
                ]
                if torch.cuda.is_available()
                else None,
                self.node.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=self.config_manager.has_unused_parameters,
            )
        else:
            self.net_wrapper = self.net
        self.local_step = torch.tensor(self.local_step, dtype=torch.int).to(self.device)
        if self.config_manager.use_jit:
            example_data = torch.from_numpy(
                NetworkDataset(
                    self.config_manager, self.dataset_base.adapter
                ).get_next_batch()
            ).to(self.device)
            example_data_list = self.net.format_data(example_data)
            self.net_wrapper = torch.jit.trace(
                self.net_wrapper, (example_data_list)
            )
        if self.config_manager:
            pass
        if self.node.is_chief_rank:
            pass

    def do_slow(self):
        if self.slow_time > 0:
            time.sleep(self.slow_time)

    def _do_train(self):
        pass

    def run(self):
        self._do_train()

    def get_sample_consume_speed(self, batch_size, step_train_times, scale=1):
        if not step_train_times:
            return 0
        if len(step_train_times) <= 1:
            return step_train_times[0]
        times = np.array(step_train_times[1:])
        speed_mean = scale * batch_size / np.mean(times)
        return speed_mean