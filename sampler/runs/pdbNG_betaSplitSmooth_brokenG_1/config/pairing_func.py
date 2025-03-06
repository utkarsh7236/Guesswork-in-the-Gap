import jax.numpy as xp


def beta_split_smooth(m1, m2, beta_1, beta_2, sep, delta_beta):
    ratio = (m2 / m1)/sep
    term1 = ratio ** -beta_1
    term2 = 0.5 * (1 + ratio**(1/delta_beta))
    power_term = (beta_1 - beta_2)/delta_beta
    ret = term1 * (term2 ** power_term)
    return ret
