import time
import socket
import os
import subprocess
import sys
import signal
import tempfile
import os
import sys

import yaml
from yaml import Loader

import psutil
default_pkg_path = os.path.dirname(os.path.abspath(__file__))
class ProcessBase:
    def __init__(
        self,
        log_file="/aiarena/logs/process_base.log",
    ) -> None:
        self.log_file = log_file
        self.proc = None

    def get_cmd_cwd(self):
        return ["echo", "123"], "/"

    # Start process
    def start(self):
        # redirect sdtout/stderr to log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        f = open(self.log_file, "w")

        # _start_model_pool
        full_cmd, cwd = self.get_cmd_cwd()
        self.proc = subprocess.Popen(
            full_cmd,
            env=os.environ,
            stderr=subprocess.STDOUT,
            stdout=f,
            bufsize=10240,
            cwd=cwd,
        )

    # Stop process
    def stop(self):
        if not self.proc:
            return

        self.proc.kill()
        if self.proc.stdout:
            self.proc.stdout.close()
        if self.proc.stderr:
            self.proc.stderr.close()

    def poll(self):
        if not self.proc:
            return
        return self.proc.poll()

    def wait(self, timeout=None):
        if not self.proc:
            return
        self.proc.wait(timeout)

    def _test_connect(self, host, port):
        with socket.socket(socket.AF_INET) as s:
            try:
                s.connect((host, port))
            except ConnectionRefusedError:
                return False
        return True

    def wait_server_started(self, host, port, timeout=-1):
        start_time = time.time()
        while timeout <= 0 or time.time() - start_time < timeout:
            if self._test_connect(host, port):
                break
            if self.proc and self.proc.poll() is not None:
                break
            time.sleep(1)

    def terminate(self):
        if self.proc:
            self.proc.terminate()

class ModelPoolProcess(ProcessBase):
    def __init__(
        self,
        role="gpu",
        master_ip="127.0.0.1",
        log_file="/aiarena/logs/model_pool.log",
        file_save_path="/mnt/ramdisk/files",
        pkg_path=default_pkg_path,
    ):
        """
        pkg_path: model_pool pkg包的路径, 用于启动程序
        log_path: model_pool 进程启动后的日志输出路径
        """
        super().__init__(log_file)
        self.pkg_path = pkg_path
        self.ip = "127.0.0.1"  # TODO 确认是否需要配置成当前ip
        self.cluster_context = "default"
        self.role = role
        self.master_ip = master_ip
        self.file_save_path = file_save_path

    def _get_config(self, role, master_ip):
        # load default config from file
        config_file = os.path.join(self.pkg_path, "config", f"trpc_go.yaml.{role}")
        with open(config_file, encoding="utf-8") as f:
            config = yaml.load(f, Loader=Loader)

        for _, log_plugin in config.get("plugins", {}).get("log", {}).items():
            for _config in log_plugin:
                if _config.get("writer_config", {}).get("filename"):
                    _config["writer_config"]["filename"] = self.log_file

        # overwrite default config
        if role == "cpu":
            config["client"]["service"][0]["target"] = f"dns://{master_ip}:10013"
            config["modelpool"]["ip"] = self.ip
            config["modelpool"]["name"] = self.ip
            config["modelpool"]["cluster"] = self.cluster_context
        elif role == "gpu":
            config["modelpool"]["ip"] = self.ip
            config["modelpool"]["fileSavePath"] = self.file_save_path
        else:
            raise Exception(f"Unknow role: {role}")
        return config

    def _generate_config_file(self, role, master_ip):
        config = self._get_config(role, master_ip)
        fd, file = tempfile.mkstemp()
        with os.fdopen(fd, "w") as f:
            yaml.dump(config, f)
        return file

    def get_exe_name(self):
        if sys.platform == "win32":
            return os.path.join(self.pkg_path, "bin", "modelpool.exe")
        else:
            return os.path.join(self.pkg_path, "bin", "modelpool")

    def get_cmd_cwd(self):
        config_file = self._generate_config_file(self.role, self.master_ip)
        full_cmd = [self.get_exe_name(), "-conf", config_file]
        cwd = os.path.join(self.pkg_path, "bin")
        return full_cmd, cwd

