import time
import arrays as np
input_matrix = np.random.sample((4, 3))
print(input_matrix)
w = np.random.random((3,1))
print(w)
b = 3

y = input_matrix.dot(w) + b
print(y > 0)


#####################################
mat = np.append(input_matrix, np.array([[1]] * input_matrix.shape[0]), axis=1)
weights = np.append(w, np.array([[b]]), axis=0)

print(mat.dot(weights) > 0)

#####################################
print(time.time())

#####################################
def sigmoid(x):
    """сигмоидальная функция, работает и с числами, и с векторами (поэлементно)"""
    return 1 / (1 + np.exp(-x))

print(sigmoid(w))

