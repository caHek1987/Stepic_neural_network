import urllib
from urllib import request
import arrays as np

fname = input()  # read file name from stdin
f = urllib.request.urlopen(fname)  # open file from URL
data = np.loadtxt(f, delimiter=',', skiprows=1)  # load data to work with

#print(data)

y = np.copy(data[:, 0])
#print(y)

data[:, 0] = 1
X = data
#print(X)

#ones = np.ones_like(data[:, 0])  # создаёт массив, состоящий из единиц, идентичный по форме массиву array
#print(ones)

#X = np.hstack((ones, data))  # склеивает по строкам массивы, являющиеся компонентами кортежа, поданного на вход; массивы должны совпадать по всем измерениям, кроме второго

b = (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(y)
#print(b)

c = map(str, b)  # применяет функцию str к каждому элементу array
print(" ".join(c))  # возвращает строку, состоящую из элементов array, разделённых символами "delim"

#####################################################

#from urllib.request import urlopen
#import numpy as np

#X = np.loadtxt(urlopen(input()), skiprows=1, delimiter=',')
#Y = X[:, 0].copy()
#X[:, 0] = 1
#print(*np.linalg.inv(X.T @ X) @ X.T @ Y)