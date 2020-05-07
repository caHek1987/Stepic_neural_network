import arrays as np

# m = 2
#
# x = np.ones((m,))
# print(x)
# print(x.shape)
#
# w = np.ones((m,))
#
# b = 0
#
# print(np.dot(w, x))

################
# input_matrix = np.ones((3, 4))
# print(input_matrix, input_matrix.shape[0])
# dy = np.ones((3, 1))
# print(dy)
# dz = np.array([[1], [2], [3]])
# print((dz*dy).T)
#
# grad = (dz * dy).T.dot(input_matrix)
#
# print(grad)
# grad - grad.T
# print(grad)

def counter_6(n):
    i = 0
    while i < n:
        print(i)
        if i == 6:
            return 1

        else:
            i += 1
    return 0

print(counter_6(7))




