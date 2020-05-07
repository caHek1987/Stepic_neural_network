import arrays as np
mat = 2 * np.eye(3,4,k=0) + np.eye(3,4,k=1)
print(mat)

c=mat.flatten()
print(c)
e = c.T
print(e)

d=mat.reshape(12,1)
print(d)