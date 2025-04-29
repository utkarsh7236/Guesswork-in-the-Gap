import jax.numpy as xp


def beta_split_3(m1, m2, beta_1, beta_2, beta_gap, sep):
    condlist = [(m1 < sep) & (m2 < sep),
                (m1 >= sep) & (m2 <= sep),
                (m1 >= sep) & (m2 >= sep)]
    choicelist = [(m2 / m1) ** beta_1, (m2 / m1) ** beta_gap, (m2 / m1) ** beta_2]
    ret = xp.select(condlist, choicelist, default=0.0)
    return ret
