from cosmology import *
import jax.numpy as xp

def powerlaw_redshift(z, mass1_source, mass2_source, H0, Om0, w, kappa):
    args = (H0, Om0, w)
    dVcdz = dVc_dz_analytic_no_dl(z, args)
    ret = dVcdz * (1+z)**(kappa-1)
    return ret
