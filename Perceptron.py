import arrays as np
import  random

class Perceptron:

    def __init__(self, w, b):
        """
        Инициализируем наш объект - перцептрон.
        w - вектор весов размера (m, 1), где m - количество переменных
        b - число
        """
        self.w = w
        self.b = b

    def forward_pass(self, single_input):
        """
        Метод рассчитывает ответ перцептрона при предъявлении одного примера
        single_input - вектор примера размера (m, 1).
        Метод возвращает число (0 или 1) или boolean (True/False)
        """
        result = 0
        for i in range(0, len(self.w)):
            result += self.w[i] * single_input[i]
        result += self.b

        if result > 0:
            return 1
        else:
            return 0

    def vectorized_forward_pass(self, input_matrix):
        """
        Метод рассчитывает ответ перцептрона при предъявлении набора примеров
        input_matrix - матрица примеров размера (n, m), каждая строка - отдельный пример,
        n - количество примеров, m - количество переменных
        Возвращает вертикальный вектор размера (n, 1) с ответами перцептрона
        (элементы вектора - boolean или целые числа (0 или 1))
        """
        return np.dot(input_matrix, self.w) + self.b > 0
        ## Этот метод необходимо реализовать

    def train_on_single_example(self, example, y):
        """
        принимает вектор активации входов example формы (m, 1)
        и правильный ответ для него (число 0 или 1 или boolean),
        обновляет значения весов перцептрона в соответствии с этим примером
        и возвращает размер ошибки, которая случилась на этом примере до изменения весов (0 или 1)
        (на её основании мы потом построим интересный график)
        """
        error = y - self.vectorized_forward_pass(example.T)
        self.w += error * example
        self.b += error
        return error
        ## Этот метод необходимо реализовать

    def train_until_convergence(self, input_matrix, y, max_steps=1e8):
        """
        input_matrix - матрица входов размера (n, m),
        y - вектор правильных ответов размера (n, 1) (y[i] - правильный ответ на пример input_matrix[i]),
        max_steps - максимальное количество шагов.
        Применяем train_on_single_example, пока не перестанем ошибаться или до умопомрачения.
        Константа max_steps - наше понимание того, что считать умопомрачением.
        """
        i = 0
        errors = 1
        while errors and i < max_steps:
            i += 1
            errors = 0
            for example, answer in zip(input_matrix, y):
                example = example.reshape((example.size, 1))
                error = self.train_on_single_example(example, answer)
                errors += int(error)  # int(True) = 1, int(False) = 0, так что можно не делать if


w = np.random.randint(1, 10, (2, 1))
b = random.randint(1, 10)
neuron1 = Perceptron(w, b)
examples = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y = np.array([0, 1, 1, 1])[:, np.newaxis]
neuron1.train_until_convergence(examples, y)

w = np.random.randint(1, 10, (2, 1))
b = random.randint(1, 10)
neuron2 = Perceptron(w, b)
y = np.array([1, 1, 1, 0])[:, np.newaxis]
neuron2.train_until_convergence(examples, y)

w = np.random.randint(1, 10, (2, 1))
b = random.randint(1, 10)
examples = np.array([[0, 1], [1, 1], [1, 0]])
neuron3 = Perceptron (w, b)
y = np.array([0, 1, 0])[:, np.newaxis]
neuron3.train_until_convergence(examples, y)

print("нейрон 1\n", neuron1.w, neuron1.b)
print("нейрон 2\n", neuron2.w, neuron2.b)
print("нейрон 3\n", neuron3.w, neuron3.b)
print(str(float(np.round(neuron1.w, 4)[0]))+',', str(float(np.round(neuron2.w, 4)[0]))+',', str(float(np.round(neuron1.w, 4)[1]))+',',
      str(float(np.round(neuron2.w, 4)[1]))+',', str(float(np.round(neuron1.b, 4)))+',',
      str(float(np.round(neuron2.b, 4)))+',', str(float(np.round(neuron3.w, 4)[0]))+',',
      str(float(np.round(neuron3.w, 4)[1]))+',', str(float(np.round(neuron3.b, 4))))