import os
import torch
import time
from multiprocessing import Process


class ModelManagerBase(object):
    def save_checkpoint(self, net, optimizer, step: int):
        pass

    def _save_checkpoint(self, net, optimizer, checkpoint_dir: str, step: int, save_optimizer_state: bool = True):
        pass

class ModelManager(ModelManagerBase):
    def __init__(self, package_code_path, actor_model_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()
        self.package_code_path = package_code_path
        self.actor_model_path = actor_model_path

    def _backup_checkpoint(self):
        pass