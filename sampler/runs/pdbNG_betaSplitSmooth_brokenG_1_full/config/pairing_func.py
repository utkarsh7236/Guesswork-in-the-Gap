import jax.numpy as xp


def logistic(x, sep, L1, L2, delta_beta, width = 20):
    scale = (x - sep) / width
    transition = 1 / (1 + xp.exp(-delta_beta * scale))
    return L1 * (1-transition) + L2 * transition

def beta_split_smooth(m1, m2, beta_1, beta_2, sep, delta_beta):
    plaw1 = (m2/m1) ** beta_1
    plaw2 = (m2/m1) ** beta_2
    ret = logistic(m2, sep, plaw1, plaw2, delta_beta)
    return ret
