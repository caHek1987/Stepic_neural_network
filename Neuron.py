# Реализуйте метод update_mini_batch класса Neuron. Когда вы решите сдать задачу, вам нужно будет просто скопировать соответствующие функции (которые вы написали в ноутбуке ) сюда. Копируем без учёта отступов; шаблон в поле ввода ответа уже будет, ориентируйтесь по нему. Сигнатура функции указана в ноутбуке, она остаётся неизменной.
# update_mini_batch считает градиент и обновляет веса нейрона на основе всей переданной ему порции данных, кроме того, возвращает 1, если алгоритм сошелся (абсолютное значение изменения целевой функции до и после обновления весов < eps), иначе возвращает 0.
# Мы будем проверять ваш алгоритм на данных разного размера. Пример данных, на которых вы можете проверить работу своего решения самостоятельно:
import arrays as np

## Определим разные полезные функции
def sigmoid(x):
    return 1 / (1 + np.exp(-x))     #  сигмоидальная функция, работает и с числами, и с векторами (поэлементно)

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))    # производная сигмоидальной функции, работает и с числами, и с векторами (поэлементно)

def J_quadratic(neuron, X, y):
    """
    Оценивает значение квадратичной целевой функции.
    Всё как в лекции, никаких хитростей.
    neuron - нейрон, у которого есть метод vectorized_forward_pass, предсказывающий значения на выборке X
    X - матрица входных активаций (n, m)
    y - вектор правильных ответов (n, 1)
    """
    assert y.shape[1] == 1, 'Incorrect y shape'
    return 0.5 * np.mean((neuron.vectorized_forward_pass(X) - y) ** 2)      # Возвращает значение J (число)

def J_quadratic_derivative(y, y_hat):
    """
    Вычисляет вектор частных производных целевой функции по каждому из предсказаний.
    y_hat - вертикальный вектор предсказаний,
    y - вертикальный вектор правильных ответов,

    В данном случае функция смехотворно простая, но если мы захотим поэкспериментировать
    с целевыми функциями - полезно вынести эти вычисления в отдельный этап.

    Возвращает вектор значений производной целевой функции для каждого примера отдельно.
    """
    assert y_hat.shape == y.shape and y_hat.shape[1] == 1, 'Incorrect shapes'

    return (y_hat - y) / len(y)

def compute_grad_analytically(neuron, X, y, J_prime=J_quadratic_derivative):
    """
    Аналитическая производная целевой функции
    neuron - объект класса Neuron
    X - вертикальная матрица входов формы (n, m), на которой считается сумма квадратов отклонений
    y - правильные ответы для примеров из матрицы X
    J_prime - функция, считающая производные целевой функции по ответам

    Возвращает вектор размера (m, 1)
    """
    # Вычисляем активации
    # z - вектор результатов сумматорной функции нейрона на разных примерах

    z = neuron.summatory(X)
    y_hat = neuron.activation(z)

    # Вычисляем нужные нам частные производные
    dy_dyhat = J_prime(y, y_hat)
    dyhat_dz = neuron.activation_function_derivative(z)

    # осознайте эту строчку:
    dz_dw = X

    # а главное, эту:
    grad = ((dy_dyhat * dyhat_dz).T).dot(dz_dw)

    # можно было написать в два этапа. Осознайте, почему получается одно и то же
    # grad_matrix = dy_dyhat * dyhat_dz * dz_dw
    # grad = np.sum(, axis=0)

    # Сделаем из горизонтального вектора вертикальный
    grad = grad.T
    return grad

class Neuron:

    def __init__(self, weights, activation_function=sigmoid, activation_function_derivative=sigmoid_prime):
        # weights - вертикальный вектор весов нейрона формы (m, 1), weights[0][0] - смещение
        # activation_function - активационная функция нейрона, сигмоидальная функция по умолчанию
        # activation_function_derivative - производная активационной функции нейрона

        assert weights.shape[1] == 1, "Incorrect weight shape"

        self.w = weights
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

    def forward_pass(self, single_input):
        """
        активационная функция логистического нейрона
        single_input - вектор входов формы (m, 1),
        первый элемент вектора single_input - единица (если вы хотите учитывать смещение)
        """
        result = 0
        for i in range(self.w.size):
            result += float(self.w[i] * single_input[i])
        return self.activation_function(result)

    def summatory(self, input_matrix):
        """
        Вычисляет результат сумматорной функции для каждого примера из input_matrix.
        input_matrix - матрица примеров размера (n, m), каждая строка - отдельный пример,
        n - количество примеров, m - количество переменных.
        Возвращает вектор значений сумматорной функции размера (n, 1).
        """
        return np.dot(input_matrix, self.w)

    def activation(self, summatory_activation):
        """
        Вычисляет для каждого примера результат активационной функции,
        получив на вход вектор значений сумматорной функций
        summatory_activation - вектор размера (n, 1),
        где summatory_activation[i] - значение суммматорной функции для i-го примера.
        Возвращает вектор размера (n, 1), содержащий в i-й строке
        значение активационной функции для i-го примера.
        """
        return self.activation_function(summatory_activation)

    def vectorized_forward_pass(self, input_matrix):
        """
        Векторизованная активационная функция логистического нейрона.
        input_matrix - матрица примеров размера (n, m), каждая строка - отдельный пример,
        n - количество примеров, m - количество переменных.
        Возвращает вертикальный вектор размера (n, 1) с выходными активациями нейрона
        (элементы вектора - float)
        """
        return self.activation(self.summatory(input_matrix))

    def SGD(self, X, y, batch_size, learning_rate=0.1, eps=1e-6, max_steps=200):
        """
        Внешний цикл алгоритма градиентного спуска.
        X - матрица входных активаций (n, m)
        y - вектор правильных ответов (n, 1)

        learning_rate - константа скорости обучения
        batch_size - размер батча, на основании которого
        рассчитывается градиент и совершается один шаг алгоритма

        eps - критерий остановки номер один: если разница между значением целевой функции
        до и после обновления весов меньше eps - алгоритм останавливается.
        Вторым вариантом была бы проверка размера градиента, а не изменение функции,
        что будет работать лучше - неочевидно. В заданиях используйте первый подход.

        max_steps - критерий остановки номер два: если количество обновлений весов
        достигло max_steps, то алгоритм останавливается

        Метод возвращает 1, если отработал первый критерий остановки (спуск сошёлся)
        и 0, если второй (спуск не достиг минимума за отведённое время).
        """
        rows_list = np.arange(0, X.shape[0], 1)

        i = 0
        while i < max_steps:
            batch_rows = np.random.choice(rows_list, batch_size, replace=False)
            # X_rows_shuffle = np.random.shuffle(X_rows)

            if self.update_mini_batch(X[batch_rows, :], y[batch_rows, :], learning_rate, eps) == 1:
                return 1
            else:
                i += 1
        return 0


        # Этот метод необходимо реализовать

    def update_mini_batch(self, X, y, learning_rate, eps):
        """
        X - матрица размера (batch_size, m)
        y - вектор правильных ответов размера (batch_size, 1)
        learning_rate - константа скорости обучения
        eps - критерий остановки номер один: если разница между значением целевой функции
        до и после обновления весов меньше eps - алгоритм останавливается.

        Рассчитывает градиент (не забывайте использовать подготовленные заранее внешние функции)
        и обновляет веса нейрона. Если ошибка изменилась меньше, чем на eps - возвращаем 1,
        иначе возвращаем 0.
        """
        J_1 = J_quadratic(self, X, y)
        grad_J = compute_grad_analytically(self, X, y)
        self.w += -learning_rate * grad_J
        J_2 = J_quadratic(self, X, y)

        return int(eps > abs(J_1 - J_2))

        # Этот метод необходимо реализовать

np.random.seed(42)
n = 10
m = 5

X = 20 * np.random.sample((n, m)) - 10
#print(X)
#print(X.shape[0])
y = (np.random.random(n) < 0.5).astype(np.int)[:, np.newaxis]
#print(y.shape)
w = 2 * np.random.random((m, 1)) - 1
#print(w.shape)

neuron = Neuron(w)
neuron.update_mini_batch(X, y, 0.1, 1e-5)

# Если вы посмотрите на веса нейрона neuron после выполнения этого кода, то они должны быть такими:
print(neuron.w)
# [[-0.22368982]
#  [-0.45599204]
#  [ 0.65727411]
#  [-0.28380677]
#  [-0.43011026]]


