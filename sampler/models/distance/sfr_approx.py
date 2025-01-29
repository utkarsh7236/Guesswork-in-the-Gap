from cosmology import *

def sfr_approx(d_l, H0, Om0, w, kappa, z_min, z_max):
    args = (H0, Om0, w)
    z = redshift_func_jax(d_l, z_min, z_max, args)
    dVc_dz = dVc_dz_analytic(d_l, z, args)
    ddl_dz = dDl_dz_analytic(d_l, z, args)
    num = (1 + z) ** kappa
    dem = 1 + ((1 + z) / 2.9) ** 5.6
    phi = 0.015 * num / dem
    ret = (dVc_dz/(1 + z)) * (phi/xp.abs(ddl_dz))
    return ret