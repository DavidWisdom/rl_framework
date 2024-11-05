import os
import torch
import time
import datetime
import shutil
import tempfile
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

    def _backup_checkpoint(
        self, save_model_dir, backup_checkpoint_dir, step, touch_done=True
    ):
        os.makedirs(backup_checkpoint_dir, exist_ok=True)
        os.makedirs(save_model_dir, exist_ok=True)
        if touch_done:
            pass
        else:
            checkpoint_file = os.path.join(save_model_dir, "model.pth")
            filename = "code_{}_{}.pth".format(
                str(datetime.datetime.now())
                .replace(" ", "_")
                .replace("-", "")
                .replace(":", ""),
                step,
            )
            ret = os.path.join(backup_checkpoint_dir, filename)
            shutil.copy(checkpoint_file, ret)
            return ret