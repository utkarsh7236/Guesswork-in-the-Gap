import jax.numpy as xp


def beta_split(m1, m2, beta_1, beta_2, sep):
    ret = xp.where(m2 < sep, (m2 / m1) ** beta_1, 0)
    ret = xp.where(m2 >= sep, (m2 / m1) ** beta_2, ret)
    return ret
