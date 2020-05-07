import numpy as np

test_data = [0, 1, 2]
for y in test_data:
    y = np.eye(3, 1, k=-int(y))
    print(y)

sizes = np.random.rand(3, 4)
print(sizes)
print(sizes[1:])