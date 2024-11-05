from numpy.core.numeric import True_


class DimConfig:
    pass

class Config:
    NETWORK_NAME = "network"

    INIT_LEARNING_RATE_START = 0.0001
    BETA_START = 0.025

    CLIP_PARAM = 0.2
    EVAL_FREQ = 5
    GAMMA = 0.995
    LAMDA = 0.95
    IS_TRAIN = True
