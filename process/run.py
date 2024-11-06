import sys
import time
import os

from multiprocessing import Process
from code.actor.entry import run as actor_run
from code.actor.entry import get_config_definition as actor_config_definition
from code.learner.train import run as learner_run
from code.learner.train import get_config_definition as learner_config_definition



def test_actor(config):
    config.single_test = True
    config.max_episode = 1
    config.max_frame_num = 1000
    process = Process(target=actor_run, args=(config,))
    process.daemon = True
    process.run()

def test_learner(config):
    pass

def start_actor(config):
    pass

def start_learner(config):
    pass



if __name__ == '__main__':
    def main(_):
        pass

