import torch
import numpy as np

# Directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)

# From a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

# retains the properties of x_data
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

# overrides the datatype of x_data
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

# torch tensor type
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# torch tensor attr
tensor = torch.rand(3, 4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# move tensor to cuda
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

# operator
print(f"\n\n")
print("tensor operator:")
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
print(f"y1 = {y1}")
print(f"y2 = {y2}")

y3 = torch.rand_like(y1)
print(f"y3 = {y3}")
torch.matmul(tensor, tensor.T, out=y3)
print(f"y3' = {y3}")

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
print(f"z1 = {z1}")
z2 = tensor.mul(tensor)
print(f"z2 = {z2}")

z3 = torch.rand_like(tensor)
print(f"z3 = {z3}")
torch.mul(tensor, tensor, out=z3)
print(f"z3' = {z3}")

a1 = torch.arange(3).reshape((3, 1))
b1 = torch.arange(2).reshape((1, 2))
print(f"a1 = {a1}")
print(f"b1 = {b1}")
c1 = a1 + b1
print(f"c1 = {c1}")


# a = torch.arange(60).reshape((3, 4, 5))
# b = torch.arange(60).reshape((5, 3, 4))
# print(f"a = {a}")
# print(f"b = {b}")
# c = a + b
# print(f" c = {c}")

print(f"\n求和：")
x = torch.arange(60).reshape((5, 4, 3))
x, x.sum()
print(f"x = {x}")
print(f"x_sum = {x.sum()}")
x_sum_axis0 = x.sum(axis=0)
print(f"x_sum_axis0 = {x_sum_axis0}")
x_sum_axis1 = x.sum(axis=1)
print(f"x_sum_axis1 = {x_sum_axis1}")
x_sum_axis2 = x.sum(axis=2)
print(f"x_sum_axis2 = {x_sum_axis2}")


