import jax.numpy as xp


def logistic(m2, sep, L1, L2, mwidth):
    scale = (m2 - sep) / mwidth
    transition = 1 / (1 + xp.exp(-scale))
    return L1 * (1-transition) + L2 * transition

def beta_split_smooth(m1, m2, beta_1, beta_2, sep, mwidth):
    plaw1 = (m2/m1) ** beta_1
    plaw2 = (m2/m1) ** beta_2
    ret = logistic(m2, sep, plaw1, plaw2, mwidth)
    return ret