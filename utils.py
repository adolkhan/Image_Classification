import os

import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn

OPTIMIZER_NAME_TO_OPTIMIZER_CLASS = {
    "Adam": optim.Adam
}

LOSS_NAME_TO_LOSS_CLASS = {
    "CrossEntropy": nn.CrossEntropyLoss
}


def create_optimizer(optimizer, model_params, **params):
    return OPTIMIZER_NAME_TO_OPTIMIZER_CLASS[optimizer](model_params, **params)


def available_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_loss(loss, **params):
    return LOSS_NAME_TO_LOSS_CLASS[loss](**params)


def set_determenistic(seed=404, determenistic=True):
    if determenistic:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.determenistic = determenistic


