#Алгоритм обратного распространения ошибки

import arrays as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))     #  сигмоидальная функция, работает и с числами, и с векторами (поэлементно)

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))    # производная сигмоидальной функции, работает и с числами, и с векторами (поэлементно)

def max(x):         # функция максимума
    if x > 0: return x
    else: return 0

def max_prime(x):       # производная функции максимума
    if x > 0: return 1
    else: return 0

# задаем данные
weights0_1 = np.array([[0.7, 0.2, 0.7], [0.8, 0.3, 0.6]])    # задаем матрицу весво между слоями 0 и 1
weights1_2 = np.array([[0.2, 0.4]])    # задаем матрицу весво между слоями 1 и 2
print("weights0_1\n", weights0_1, "\nweights1_2\n", weights1_2)

b1 = np.zeros((2, 1))
b2 = np.zeros((1, 1))

input_a = np.array([[0], [1], [1]])   # задаем вектор входных активаций
print("input_activations\n", input_a, input_a.shape)

y = np.ones((1,1))   # задаем вектор правильных ответов
print("answer y\n", y, y.shape)

# прямое распространение активаций
z1 = np.dot(weights0_1, input_a) + b1
a1_1 = max(z1[0][0])
a1_2 = sigmoid(z1[1][0])    # вектор активаций 1го слоя (помежуточного)
a1 = np.array([[a1_1], [a1_2]])
print("activations1\n", a1_1, a1.shape)

z2 = np.dot(weights1_2, a1) + b2
a2 = sigmoid(z2)    # вектор активаций 2го слоя (выходного)
print("activations2\n", a2, a2.shape)

# обратное распространение ошибки
deltas2 = (a2 - y) * sigmoid_prime(z2)      # считаем вектор дельт (ошибок) 2го слоя (выходного)
print("deltas2\n", deltas2, deltas2.shape)
deltas1_1 = weights1_2[0][0] * deltas2[0][0] * max_prime(z1[0][0])
deltas1_2 = weights1_2[0][1] * deltas2[0][0] * sigmoid_prime(z1[1][0])      # считаем вектор ошибок (дельт) 1го слоя (промежуточного)
deltas1 = np.array([[deltas1_1], [deltas1_2]])
print("deltas1\n", deltas1, deltas1.shape)

grad_Jb2 = deltas2      # считаем градиент целевой функции по смещениям для 2го слоя (выходного)
grad_Jb1 = deltas1
print("gradient dJ_db2\n", grad_Jb2, grad_Jb2.shape, "\ngradient dJ_db1\n", grad_Jb1, grad_Jb1.shape)
grad_Jw1_2 = np.dot(a1, deltas2.T)
grad_Jw0_1 = np.dot(input_a, deltas1.T)      # считаем градиент целевой функции по весам для 1го слоя (промежуточного)
print("gradient dJ_dw12\n", grad_Jw1_2, grad_Jw1_2.shape, "\ngradient dJ_dw01\n", grad_Jw0_1, grad_Jw0_1.shape)

print(grad_Jw0_1[2][0], grad_Jw0_1[2][1])