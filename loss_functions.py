import arrays as np
import math


def J_quadratic(a, y):   #квадратичная целевая функция
    # a - вектор входных активаций (n, 1)
    # y - вектор правильных ответов (n, 1)
    return 0.5 * np.mean((a - y) ** 2)      # Возвращает значение J (число)


def J_cross_entopy(a, y):   # кросс энтропийная целевая функция
    J = 0
    for i in range(len(a-1)):
        J += (y[i] * ln(a[i]) + (1-y[i]) * ln(1-a[i]))
    return -1 * J / len(a)


def ln(x):
    if x == 0:
        return 0
    else:
        return math.log(x)


a = np.array([1, 0.3, 0.1])
y = np.array([1, 0, 0])

print(J_quadratic(a, y), J_cross_entopy(a, y))
