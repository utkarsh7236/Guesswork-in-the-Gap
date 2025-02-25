from cosmology import *

def sfr_approx(z, H0, Om0, w, kappa):
    args = (H0, Om0, w)
    dVc_dz = dVc_dz_analytic_no_dl(z, args)
    ddl_dz = dDl_dz_analytic_no_dl(z, args)
    num = (1 + z) ** kappa
    dem = 1 + ((1 + z) / 2.9) ** 5.6
    phi = 0.015 * num / dem
    ret = (dVc_dz/(1 + z)) * (phi/xp.abs(ddl_dz))
    return ret