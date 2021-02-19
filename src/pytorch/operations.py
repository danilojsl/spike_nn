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


def sum_tensors():
    tensor1 = torch.rand(2, 3)
    print(tensor1)
    tensor2 = torch.rand(2, 3)
    print(tensor2)

    plus_result = tensor1 + tensor2
    print(plus_result)


def autograd():
    x = torch.ones(1, requires_grad=True)
    # Prints None since there is nothing to be calculated
    print(x.grad)


def working_autograd():
    x = torch.ones(1, requires_grad=True)
    y = x + 2
    z = y * y * 2
    # The gradient this time is calculated since z is a function of y and y a function of x
    z.backward()    # automatically calculates the gradient
    print(x.grad)   # ∂z/∂x = 12


if __name__ == '__main__':
    sum_tensors()
    working_autograd()
