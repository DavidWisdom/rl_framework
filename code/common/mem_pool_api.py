import logging
import zmq

from code.common.mem_pool_protocol import MemPoolProtocol


class ZmqSocket:
    def __init__(self, ip_port, sock_type="client"):
        self.ip_port = ip_port
        self.timeout = 1000 * 30  # ms
        self.context = zmq.Context()
        self.socket = None
        self.poller_send = zmq.Poller()
        self.poller_recv = zmq.Poller()
        self._connect()

    def _connect(self):
        if self.socket:
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.close()
            self.poller_send.unregister(self.socket)
            self.poller_recv.unregister(self.socket)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.ip_port)
        self.poller_send.register(self.socket, zmq.POLLOUT)
        self.poller_recv.register(self.socket, zmq.POLLIN)

    def syn_send_recv(self, message):
        while True:
            if self.poller_send.poll(self.timeout):
                self.socket.send(message)
            else:
                logging.error("send timeout, try to reconnect")
                self._connect()
                continue

            if self.poller_recv.poll(self.timeout):
                data = self.socket.recv()
                break
            else:
                logging.error("recv timeout, try to reconnect")
                self._connect()
                continue
        return data

    def syn_recv_send(self, message):
        msg = self.syn_recv()
        self.syn_send(message)
        return msg

    def syn_recv(self):
        while True:
            socks = self.poller_recv.poll(self.timeout)
            # print(socks, type(socks))
            if socks:
                data = self.socket.recv()
                break
            else:
                logging.error("recv timeout, try to reconnect")
                self._connect()
        return data

    def syn_send(self, message):
        while True:
            socks = self.poller_send.poll(self.timeout)
            # print(socks, type(socks))
            if socks:
                self.socket.send(message)
                break
            else:
                logging.error("send timeout, try to reconnect")
                self._connect()
import struct
import socket
import logging
import time


class TcpSocket:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = int(port)
        self.sock = None
        self._connect(self.ip, self.port)

    def _connect(self, ip, port):
        address = (ip, port)
        logging.info("address:%s" % str(address))
        while True:
            try:
                if self.sock:
                    self.sock.close()
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect(address)
                return True
            except Exception as e:  # pylint: disable=broad-except
                logging.error("connect failed, address:%s, except:%s" % (address, e))
                time.sleep(1)

    def _send_all(self, request):
        try:
            _ = self.sock.sendall(request)
            return True
        except Exception as e:  # pylint: disable=broad-except
            logging.error("send failed, except:%s" % e)
            return False

    def _recv_all(self, recv_len):
        recved_len = 0
        recv_data = b""
        while recved_len < recv_len:
            try:
                data = self.sock.recv(recv_len - recved_len)
            except Exception as e:  # pylint: disable=broad-except
                logging.error("recv failed, except:%s" % e)
                return False, None
            if data == b"":
                logging.error("recv failed, data is empty")
                return False, None
            recv_data = recv_data + data
            recved_len += len(data)

        if recved_len != recv_len:
            logging.error("recv failed, recved_len != recv_len")
            return False, recv_data
        else:
            return True, recv_data

    def syn_send_recv(self, request):
        ret = True
        while True:
            # check status
            if not ret:
                logging.error("conn is error, try to reconnect")
                self._connect(self.ip, self.port)
                time.sleep(1)

            # send request
            ret = self._send_all(request)
            if not ret:
                logging.error("_send_all failed")
                continue

            # recv header
            head_length = 4
            ret, recv_data = self._recv_all(head_length)
            if not ret:
                logging.error("_recv_all data_len failed")
                continue

            # recv proto_data
            total_len = struct.unpack("I", recv_data)[0]
            total_len = socket.ntohl(total_len)
            if total_len - head_length > 0:
                ret, proto_data = self._recv_all(total_len - head_length)
                recv_data += proto_data
            if not ret:
                logging.error("_recv_all data failed")
                continue

            return recv_data

import struct
import socket
from enum import Enum
import lz4.block


class MemPoolAPIs(object):
    #  The constructor.
    #  @param self The object pointer.
    #  @param ip mempool server ip
    #  @param port mempool server port
    #  @param socket_type mempool server type,
    #         "zmq": mempool is a python version, use zeromq protocol
    #         "mcp++": mempool is a mcp++ version, use tcp protocol
    def __init__(self, ip, port, socket_type="zmq"):
        if socket_type == "zmq":
            ip_port = "tcp://%s:%s" % (ip, port)
            self._client = ZmqSocket(ip_port, "client")
        elif socket_type == "mcp++":
            self._client = TcpSocket(ip, port)
        else:
            raise NotImplementedError

        self.protocol = MemPoolProtocol()

    #  Pull sample Interface: randomly pull a sample from mempool
    #  @param self The object pointer.
    #  @param strategy sampling strategy type:int.
    #  @return seq sequence number
    #  @return sample
    def pull_sample(self, strategy):
        request = self.protocol.format_get_request(strategy=strategy)
        response = self._request(request)
        _, seq, _, sample = self.protocol.parse_get_response(response)
        return seq, sample

    #  Push samples Interface:
    #       compress each sample by lz4 and send to mempool
    #       if more than max_sample_num, split to packages, one package include max_sample_num samples
    #  @param self The object pointer.
    #  @param samples type:list, sample type:str or bytes
    #  @param priorities type:list, priority type:float
    #  @param max_sample_num max_sample_num type:int, default 128
    #  @return ret_array, success or fail
    def push_samples(self, samples, priorities=None, max_sample_num=128):
        format_samples = self.protocol.format_batch_samples_array(
            samples, priorities, max_sample_num
        )
        ret_array = []
        for format_sample in format_samples:
            ret = self._request(format_sample)
            ret_array.append(ret)
        return ret_array

    def clean_samples(self):
        request = self.protocol.format_clean_request()
        response = self._request(request)
        return response

    def _request(self, data):
        return self._client.syn_send_recv(data)