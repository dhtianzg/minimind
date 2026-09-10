import torch

dtype = torch.float
# 本次计算在cpu上进行
device = torch.device("cpu")

# a = torch.randn(2,2,2,5, dtype=dtype, device=device)
# b = torch.randn(2,3, dtype=dtype, device=device)
a = torch.ones([2,3])


print(a)
print(a.dim())
print(a.shape)
# print(a.t())

# tesnor_grad = torch.tensor([1.0], requires_grad=True)
# tensor_result = tesnor_grad * 2
# z = tesnor_grad * tesnor_grad *3
# out = z.mean()
# print(out)