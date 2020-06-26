import os

from utils.load_config import get_attribute

import torch


# convert data from cpu to gpu, accelerate the running speed
def convert_to_gpu(data):
    if get_attribute('cuda') != -1 and torch.cuda.is_available():
        data = data.cuda(get_attribute('cuda'))
    return data


def convert_train_truth_to_gpu(train_data, truth_data):
    train_data = [convert_to_gpu(data) for data in train_data]
    # truth_data = [convert_to_gpu(data) for data in truth_data]
    truth_data = convert_to_gpu(truth_data)
    return train_data, truth_data


# maxPool on the input tensor, in the item dimension, return the pooled value
def maxPool(tensor, dim):
    return torch.max(tensor, dim)[0]


# avgPool on the input tensor, in the item dimension
def avgPool(tensor, dim):
    return torch.mean(tensor, dim)


# load parameters of model
def load_model(model, modelFilePath):
    model.load_state_dict(torch.load(modelFilePath))
    return model


# saves parameters of the model
def save_model(path: str, **save_dict):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    torch.save(save_dict, path)
