# Напишите функцию, которая, используя набор ошибок delta^{l+1} для n примеров, матрицу весов W^{l+1} и набор
# значений сумматорной функции на l-м шаге для этих примеров, возвращает значение ошибки delta^l на l-м слое сети.
# Все нейроны в сети — сигмоидальные. Функции sigmoid и sigmoid_prime уже определены.

import arrays as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))     #  сигмоидальная функция, работает и с числами, и с векторами (поэлементно)

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))    # производная сигмоидальной функции, работает и с числами, и с векторами (поэлементно)

def get_error(deltas, sums, weights):
    """
    compute error on the previous layer of network
    deltas - ndarray of shape (n, n_{l+1})
    sums - ndarray of shape (n, n_l)
    weights - ndarray of shape (n_{l+1}, n_l)
    """
    return np.mean(np.dot(deltas, weights) * sigmoid_prime(sums), axis=0).T

deltas = np.random.random((2, 3))
sums = np.random.random((2, 4))
weights = np.random.random((3, 4))
print("deltas\n", deltas, "\n", "sums\n", sums, "\n", "weights\n", weights)

a = np.dot(deltas, weights)
print("deltas.dot(weights)\n", a)
b = a * sigmoid_prime(sums)
print("*sigmoid_prime(sums)\n", b)
c = np.mean(b, axis=0)
print("average delta\n", c)
d = c.T
print("delta = average transpose\n", d)

