import multiprocessing

import lz4.block

import torch
import numpy as np

from code.common.zmq_mem_pool import ZMQMEMPOOL
from code.common.batch_process import BatchProcess
from code.common.membuffer import MemBuffer


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
from abc import abstractmethod


class NetworkDatasetBase(object):
    def __init__(self, config_manager, adapter):
        raise NotImplementedError("build model: not implemented!")

    @abstractmethod
    def get_next_batch(self):
        raise NotImplementedError("build model: not implemented!")

    def get_recv_speed(self):
        return None

class NetworkDatasetRandom(NetworkDatasetBase):
    def __init__(self, config_manager, adapter):
        super().__init__(config_manager, adapter)
        self.use_fp16 = config_manager.use_fp16
        self.batch_size = config_manager.batch_size
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

class NetworkDatasetZMQ(NetworkDatasetBase):
    def __init__(self, config_manager, adapter, port=35200):
        super().__init__(config_manager, adapter)
        self.max_sample = config_manager.max_sample
        self.batch_size = config_manager.batch_size
        self.adapter = adapter
        self.data_shapes = self.adapter.get_data_shapes()
        self.use_fp16 = config_manager.use_fp16
        self.membuffer = MemBuffer(
            config_manager.max_sample, self.data_shapes[0][0], self.use_fp16
        )

        self.batch_process = BatchProcess(
            self.batch_size,
            self.data_shapes[0][0],
            config_manager.batch_process,
            self.use_fp16,
        )

        self.port = port
        self.zmq_mem_pool = ZMQMEMPOOL(self.port)
        self.init_dataset = False

        for i in range(config_manager.sample_process):
            pid = multiprocessing.Process(target=self.enqueue_data, args=(i,))
            pid.daemon = True
            pid.start()

        self.batch_process.process(self.membuffer.get_sample)
        self.last_batch_index = -1

    def get_next_batch(self):
        batch_index, sample_buf = self.batch_process.get_batch_data()
        if self.last_batch_index >= 0:
            self.batch_process.put_free_data(self.last_batch_index)
        self.last_batch_index = batch_index

        return sample_buf

    def enqueue_data(self, process_index):
        while True:
            for sample in self.zmq_mem_pool.pull_samples():
                decompress_data = lz4.block.decompress(
                    sample, uncompressed_size=3 * 1024 * 1024
                )
                sample_list = self.adapter.deserialization(decompress_data)
                for sample in sample_list:
                    self.membuffer.append(sample)

    def get_recv_speed(self):
        return self.membuffer.get_speed()
