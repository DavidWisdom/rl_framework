import os


class DimConfig:
    pass

class Config:
    actor_num = int(os.getenv("ACTOR_NUM", "1"))
    auto_bind_cpu = os.getenv("AUTO_BIND_CPU", "0") == "1"

    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 4
    LSTM_UNIT_SIZE = 16


    SERI_VEC_SPLIT_SHAPE = [(18,), (10,)]
    INIT_LEARNING_RATE_START = 0.0001
    BETA_START = 0.025
    LOG_EPSILON = 1e-6
    LABEL_SIZE_LIST = [10]
    IS_REINFORCE_TASK_LIST = [
        True
    ]

    RMSPROP_DECAY = 0.9
    RMSPROP_MOMENTUM = 0.0
    RMSPROP_EPSILON = 0.01
    CLIP_PARAM = 0.2

    MIN_POLICY = 0.00001
    TASK_ID = 15428
    TASK_UUID = "a2dbb49f-8a67-4bd4-9dc5-69e78422e72e"

    TARGET_EMBED_DIM = 32
    data_keys = (
        "observation,reward,advantage,"
        "label0,"
        "prob0,"
        "weight0,"
        "is_train, lstm_cell, lstm_hidden_state"
    )
    data_shapes = [
        # TODO:

    ]
    key_types = (
        "tf.float32,tf.float32,tf.float32,"
        "tf.int32,"
        "tf.float32,"
        "tf.float32,"
        "tf.float32,tf.float32,tf.float32"
    )
    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    EVAL_FREQ = 5
    GAMMA = 0.995
    LAMDA = 0.95
    IS_TRAIN = True
