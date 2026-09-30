import numpy as np

# Input image matrix
image = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])

# Kernel matrix
kernel = np.array([
    [-1, -1, -1],
    [0,  0,  0],
    [1,  1,  1]
])

# Output feature map size
output_size = 3

# Create empty feature map
feature_map = np.zeros((output_size, output_size))

# Perform convolution
for i in range(output_size):
    for j in range(output_size):

        # Select 3x3 region
        region = image[i:i+3, j:j+3]

        # Multiply region with kernel and add
        feature_map[i][j] = np.sum(region * kernel)

print("Input Image:")
print(image)

print("\nKernel:")
print(kernel)

print("\nFeature Map:")
print(feature_map)