n_M = 7822 # число заразившихся на 10.04.2020
k_M = 1124/7822 # каждый день добавляется примерно 20% зараженных от общего числа
print("коэффициент заболеваемочти в Москве = ", k_M)
n_R = 11917
k_R = 1786/11917
print("коэффициент заболеваемочти в России = ", k_R)

for t in range(69):
    print(t+10, "апреля ")
    print("количество заболевших в Москве ", n_M)
    n_M = n_M + n_M * 0.15

    print("количество заболевших в России ", n_R)
    n_R = n_R + n_R * 0.15
