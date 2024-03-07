# import torch

# # Create a tensor
# tensor = torch.ones(3, 4)  # Shape: (3, 4)

# # Define indices to update
# index = torch.tensor([0, 2, 1])  # Indices along the 0th dimension

# # Values to add at the specified indices
# source = torch.tensor([1., 2., 3.])

# # Perform index_add_ operation
# tensor.index_put_(index, source)

# print(tensor)

import torch

# Create a tensor
tensor = torch.zeros(3, 4)  # Shape: (3, 4)

# Define indices to modify
indices = (
    torch.tensor([0, 2, 1]),  # Indices along the first dimension
    torch.tensor([1, 2, 3])   # Indices along the second dimension
)

# Values to be assigned at the specified indices
values = torch.tensor([1., 2., 3.])

# Perform index_put_ operation
tensor.index_put_(indices, values, accumulate=False)

print(tensor)

tensor.index_put_(indices, values, accumulate=True)

print(tensor)