from cosmology import *
import jax.numpy as xp

def powerlaw_redshift_massdep(z, mass1_source, mass2_source, H0, Om0, w, kappa1, kappa2, m_break_kappa):
    args = (H0, Om0, w)
    dVcdz = dVc_dz_analytic_no_dl(z, args)
    ret = xp.where(mass1_source < m_break_kappa, dVcdz * (1+z)**(kappa1-1), dVcdz * (1+z)**(kappa2-1))
    return ret
