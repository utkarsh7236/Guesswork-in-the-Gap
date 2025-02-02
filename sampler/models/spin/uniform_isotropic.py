from cosmology import *


def uniform_isotropic(mass1_source, mass2_source, a1, costilt1, a2, costilt2):
    a_max = 1
    a_min = 0
    costilt_max = 1
    costilt_min = -1
    p_a1 = 1 / (a_max - a_min)
    p_a2 = 1 / (a_max - a_min)
    p_costilt1 = 1 / (costilt_max - costilt_min)
    p_costilt2 = 1 / (costilt_max - costilt_min)
    p_chi1 = p_a1 * p_costilt1
    p_chi2 = p_a2 * p_costilt2
    return p_chi1 * p_chi2
