def uniform_isotropic(s1x, s1y, s1z, s2x, s2y, s2z):
    spin_1 = 1 / (s1x ** 2 + s1y ** 2 + s1z ** 2)  # 1/|s|^2
    spin_2 = 1 / (s2x ** 2 + s2y ** 2 + s2z ** 2)  # 1/|s|^2
    return spin_1 * spin_2
