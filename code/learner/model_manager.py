import os
import torch
import time
import datetime
import shutil
import tempfile
import tarfile


class ModelManagerBase(object):
    def __init__(
        self,
        save_checkpoint_dir: str = "/aiarena/checkpoints/",
        backup_checkpoint_dir: str = "/aiarena/backup/",
        push_model_queuing_dir: str = "/mnt/ramdisk/checkpoints/",
        load_optimizer_state = True,
        save_interval = 1800,
    ):
        self.save_checkpoint_dir = save_checkpoint_dir
        self.backup_checkpoint_dir = backup_checkpoint_dir
        self.push_model_queuing_dir = push_model_queuing_dir

        self.last_save_time = 0
        self.save_interval = save_interval

    def _backup_checkpoint(
        self, save_model_dir, backup_checkpoint_dir, step, touch_done=True
    ):
        os.makedirs(backup_checkpoint_dir, exist_ok=True)
        tem = (
            str(datetime.datetime.now())
            .replace(" ", "_")
            .replace("-", "")
            .replace(":", "")
        )
        temp_ckpt = f"checkpoints_{tem}_{step}"

        # 复制目录
        shutil.copytree(save_model_dir, temp_ckpt)

        # 创建tar文件
        with tarfile.open(f"{backup_checkpoint_dir}/{temp_ckpt}.tar", "w") as tar:
            tar.add(temp_ckpt)

        # 删除临时目录
        shutil.rmtree(temp_ckpt)

        if touch_done:
            done_file_path = f"{backup_checkpoint_dir}/{temp_ckpt}.tar.done"
            with open(done_file_path, "a"):
                pass
        return os.path.abspath(os.path.join(backup_checkpoint_dir, f"{temp_ckpt}.tar"))

    def save_checkpoint(self, net, optimizer, step: int):
        self._save_checkpoint(net, optimizer, self.save_checkpoint_dir, step)

    def _save_checkpoint(self, net, optimizer, checkpoint_dir: str, step: int, save_optimizer_state: bool = True):
        os.makedirs(checkpoint_dir, exist_ok=True)
        step = int(step)
        checkpoint_file = os.path.join(checkpoint_dir, "model.pth")
        if save_optimizer_state:
            torch.save(
                {
                    "network_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                },
                checkpoint_file,
            )
        else:
            torch.save(
                {
                    "network_state_dict": net.state_dict(),
                    "step": step,
                },
                checkpoint_file,
            )

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
            with tempfile.TemporaryDirectory() as tmpdirname:
                temp_work_dir = os.path.join(tmpdirname, "work")
                temp_code_dir = os.path.join(temp_work_dir, "code")

                # ignore checkpoint and __pycache__ for src code
                patterns = ["*.pth", "__pycache__"]
                ignore_pattern = shutil.ignore_patterns(*patterns)

                # copy src code to temp dir
                shutil.copytree(
                    self.package_code_path,
                    temp_code_dir,
                    ignore=ignore_pattern,
                    dirs_exist_ok=True,
                )

                # copy checkpoint to actor model path
                shutil.copytree(
                    save_model_dir,
                    os.path.join(temp_work_dir, self.actor_model_path),
                    dirs_exist_ok=True,
                )

                # zip
                filename = "code_{}_{}".format(
                    str(datetime.datetime.now())
                    .replace(" ", "_")
                    .replace("-", "")
                    .replace(":", ""),
                    step,
                )
                ret = os.path.join(backup_checkpoint_dir, filename)
                shutil.make_archive(ret, "zip", temp_work_dir)
                return ret + ".zip"
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