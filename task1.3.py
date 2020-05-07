import arrays as np

x_shape = tuple(map(int, input().split()))
X = np.fromiter(map(int, input().split()), np.int).reshape(x_shape)
y_shape = tuple(map(int, input().split()))
Y = np.fromiter(map(int, input().split()), np.int).reshape(y_shape)

print(x_shape)
print(y_shape)
print(X)
print(Y)

try:
    print(X.dot(Y.T))
except ValueError:
    print("matrix shapes do not match")

