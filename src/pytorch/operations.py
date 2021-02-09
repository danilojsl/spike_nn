import torch


def concat_two_tensors():
    tensor1 = torch.rand(2, 3)
    tensor2 = torch.rand(2, 3)
    tensors = [tensor1, tensor2]
    valid_l = [x for x in tensors if x is not None]
    print(valid_l)
    dimension = len(valid_l[0].size()) - 1
    result = torch.cat(valid_l, dimension)
    print(result.shape)
    print(result)


def concat_three_tensors():
    tensor1 = torch.rand(2, 3)
    tensor2 = torch.rand(2, 3)
    tensor3 = torch.rand(2, 3)
    tensors = [tensor1, tensor2, tensor3]
    valid_l = [x for x in tensors if x is not None]
    print(valid_l)
    dimension = len(valid_l[0].size()) - 1
    result = torch.cat(valid_l, dimension)
    print(result.shape)
    print(result)


if __name__ == '__main__':
    concat_three_tensors()
