import jax.numpy as xp


def beta_split_3(m1, m2, beta_1, beta_2, beta_gap, gamma_high, gamma_low):
    ret = xp.where(m2 < gamma_low, (m2 / m1) ** beta_1,
                   xp.where(m2 < gamma_high, (m2 / m1) ** beta_gap, (m2 / m1) ** beta_2))
    return ret
