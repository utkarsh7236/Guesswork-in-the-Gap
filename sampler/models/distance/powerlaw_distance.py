from cosmology import *
import jax.numpy as xp

def powerlaw_distance(z, H0, Om0, w, kappa):
    args = (H0, Om0, w)
    dVcdz = dVc_dz_analytic_no_dl(z, args)
    ret = dVcdz/(1+z) * (1+z)**kappa
    return ret