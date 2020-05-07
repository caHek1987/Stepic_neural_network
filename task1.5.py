import arrays as np

X = np.array([[1,60], [1,50], [1,75]])
print(X)

y = np.array([[10],[7],[12]])
print(y)

c = X.T.dot(X)
print(c)

d = np.linalg.inv(c)
print(d)

e = d.dot(X.T)
print(e)

f = e.dot(y)
print(f)