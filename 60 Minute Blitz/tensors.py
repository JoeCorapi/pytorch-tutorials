import torch
import numpy as np

# Tensor directly from array
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# Tensor from ndarray
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# Tensor from another tensor
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

# Tensor Properties
# Shape determines a tensors dimensionrs
shape = (2, 3)
rand_tensor = torch.rand(shape)
print(f"Zeros Tensor: \n {rand_tensor}")

tensor = torch.rand(3, 4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
    print(f"Device tensor is stored on: {tensor.device}")


tensor = torch.ones(4, 4)
# Tensor Indexing -> [Rows, Cols] -> All Rows, Col 1
tensor[:, 1] = 0

# Concatenate tensors along a dimension: dim0=rows, dim1=cols
t1 = torch.cat([tensor, tensor, tensor], dim=1)
t2 = torch.cat([tensor, tensor, tensor], dim=0)
print(t1)
print(t2)


# This computes the element-wise product. Multiplies directly matching elements. If elements can't be matched -> Error
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")

# This computes the matrix multiplication (dot product) between two tensors
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

# In-place operations Operations that have a _ suffix are in-place.
# Generally discouraged to use unless memory optimization is extremely important
tensor.add_(5)
print(tensor)

# Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
