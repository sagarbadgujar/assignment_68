import numpy as np

# Input feature map
feature_map = np.array([
    [3, -3, 3],
    [0,  0, 0],
    [-3, -3, -3]
])

print("Original Feature Map:")
print(feature_map)

# Apply ReLU
relu_output = np.maximum(0, feature_map)

print("\nAfter ReLU:")
print(relu_output)

# Apply 2x2 Max Pooling
pool_size = 2
stride = 2

output_size = 1
pooled_output = np.zeros((output_size, output_size))

for i in range(output_size):
    for j in range(output_size):

        region = relu_output[
            i * stride:i * stride + pool_size,
            j * stride:j * stride + pool_size
        ]

        pooled_output[i][j] = np.max(region)

print("\nAfter Max Pooling:")
print(pooled_output)