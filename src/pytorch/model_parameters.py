import torch
import torch.nn as nn


def define_parameter():
    shape = (5, 10)
    random_tensor = nn.init.xavier_uniform(torch.Tensor(*shape))
    parameter = nn.Parameter(random_tensor)
    print(parameter)


if __name__ == '__main__':
    define_parameter()
