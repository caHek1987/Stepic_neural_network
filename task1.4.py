from urllib.request import urlopen
import arrays as np

filename = input()
f = urlopen('https://stepic.org/media/attachments/lesson/16462/boston_houses.csv')

boston_houses = np.loadtxt(f, skiprows=1, delimiter=",")

#print(boston_houses)
print(boston_houses.mean(axis=0))