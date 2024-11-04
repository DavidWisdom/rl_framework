from multiprocessing import Value, Array, Queue
import random
import sys
import time
import numpy as np
import ctypes


class MemBuffer(object):
    def __init__(self, max_sample_num, sample_size, use_fp16, _max_lock_num=50000):
        self._max_len = int(max_sample_num)
        self._sample_size = sample_size
        self._use_fp16 = use_fp16
        self._max_lock_num = _max_lock_num
        if self._use_fp16:
            self._c_data_type = ctypes.c_uint16


        self.next_idx = Value('i', 0)
        self._len = Value('i', 0)
        self.recv_samples = Value('i', 0)
        self.start_time = time.time()
        self.start_sample_num = 0
        self.last_speed = 0

    def __len__(self):
        length = self._len.value
        return length

    def get_sample(self):
        error_index = 0


    def get_speed(self):
        return None


class MemQueue(object):
    def __init__(self, max_sample_num, sample_size):
        self._max_len = int(max_sample_num)
        self._sample_size = int(sample_size)
        self._data_queue = Queue(self._max_len)

    def __len__(self):
        return self._data_queue.qsize()

    def append(self, data):
        try:
            # self._data_queue.put(data, block=False)
            self._data_queue.put(data)
        except Exception as e:
            pass

    def get_sample(self):
        return self._data_queue.get()

    def clear(self):
        while not self._data_queue.empty():
            self._data_queue.get()

    def get_speed(self):
        return None
