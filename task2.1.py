import arrays as np

data = np.array([[1, 1, 0.3], [1, 0.4, 0.5], [0, 0.7, 0.8]])
print(data)

y= np.copy(data[:, 0])
print(y)

data[:, 0] = 1
X = data
print(X)

b = (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(y)
print(b)

c = map(str, b)  # применяет функцию str к каждому элементу array
print(" ".join(c))  # возвращает строку, состоящую из элементов array, разделённых символами "delim"

delta_w = (y - y_h)*x