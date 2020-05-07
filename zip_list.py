###############
a = [10, 20, 30, 40]
b = ['a', 'b', 'c', 'd', 'e']
for i, j in zip(a, b):
    print(i, j)

for i in zip(a, b):
    print(i, type(i))

a = [10, 20, 30, 40]
c = [1.1, 1.2, 1.3, 1.4]
ac = zip(a, c)
print(ac, type(ac))

ac = list(ac)
print(ac, type(ac))



