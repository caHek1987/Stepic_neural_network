import arrays as np

a = np.eye(3, 4, k=0)
print(a)
b = np.eye(3, 4, k=1)
print(b)
c = 2*a + b
print(c)