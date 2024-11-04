from multiprocessing.process import BaseProcess

import torch
import numpy as np

from code.learner.membuffer import MemBuffer


class Datasets(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def next(self):
        return torch.from_numpy(self.dataset.get_next_batch())

    def get_recv_speed(self):
        return self.dataset.get_recv_speed()

class DataPrefetch:
    def __init__(self, dataset, device, use_fp16):
        self.dataset = dataset
        self.device = device
        self.use_fp16 = use_fp16
        self.next_data = None
        self.stream = torch.cuda.Stream(device=self.device)
        self.preload()

    def preload(self):
        self.next_data = self.dataset.get_next_batch()
        with torch.cuda.stream(self.stream):
            self.next_data = torch.from_numpy(self.next_data).to(device=self.device, non_blocking=True)
            if self.use_fp16:
                self.next_data = torch.from_numpy(self.next_data).to(dtype=torch.float32, non_blocking=True)

    def next(self):
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        next_data = self.next_data
        self.preload()
        return next_data

    def get_recv_speed(self):
        return self.dataset.get_recv_speed()

class NetworkDatasetRandom(object):
    def __init__(self, config_manger, adapter):
        self.use_fp16 = config_manger.use_fp16
        self.batch_size = config_manger.batch_size
        self.adapter = adapter
        self.data_shapes = self.adapter.get_data_shapes()
        self.sample_length = self.data_shapes[0][0]
        self.sample = np.random.random([self.batch_size, self.sample_length])
        if self.use_fp16:
            self.sample = self.sample.astype(np.float16)
        else:
            self.sample = self.sample.astype(np.float32)

    def get_next_batch(self):
        return self.sample

    def get_recv_speed(self):
        return None

class NetworkDataset(object):
    def __init__(self, config_manger, adapter):
        self.max_sample = config_manger.max_sample
        self.batch_size = config_manger.batch_size
        self.adapter = adapter
        self.data_shapes = self.adapter.get_data_shapes()
        self.use_fp16 = config_manger.use_fp16
        self.mem_buffer = MemBuffer(
            config_manger.max_sample, self.data_shapes[0][0], self.use_fp16
        )
        self.batch_process =

    def get_next_batch(self):
        pass

    def enqueue_data(self):
        pass

    def get_recv_speed(self):
        pass


