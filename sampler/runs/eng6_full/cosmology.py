import jax
import jax.numpy as xp

jax.config.update("jax_enable_x64", True)


def m_source(m_det, redshift):
    return m_det / (1 + redshift)


def m_det(m_source, redshift):
    return m_source * (1 + redshift)


def luminosity_distance_manual(z, args):
    H0, Om0, w = args
    c = 299792458 / 1000  # km/s
    E_z = get_E_z(z, args)
    comoving_distance = c / H0 * xp.cumsum(1 / E_z) * (z[1] - z[0])
    ret = (1 + z) * comoving_distance
    return ret

def redshift_func_jax(d_l, zmin, zmax, args):
    _n = 100000
    _z_arr = xp.linspace(zmin, zmax, _n)
    d_l_arr = luminosity_distance_manual(_z_arr, args)
    ret = xp.interp(d_l, d_l_arr, _z_arr)
    return ret

def d_l_func_jax(z_vals, args, zmin = 0.001, zmax = 2):
    _n = 100000
    _z_arr = xp.linspace(zmin, zmax, _n)
    d_l_arr = luminosity_distance_manual(_z_arr, args)
    ret = xp.interp(z_vals, _z_arr, d_l_arr)
    return ret

def get_E_z(z, args):
    H0, Om0, w = args
    omega_M = Om0
    omega_R = 0.00149369445
    omega_k = 0
    omega_L = 1 - omega_M - omega_R - omega_k # 0.6888463055445441
    E_z = xp.sqrt(omega_M * (1 + z) ** 3 + omega_R * (1 + z) ** 4 + omega_k * (1 + z) ** 2 + omega_L * (1 + z) ** (3 * (1 + w)))
    return E_z

def dDl_dz_analytic(d_l, z, args):
    H0, Om0, w = args
    c = 299792458 / 1000  # km/s
    E_z = get_E_z(z, args)

    term1 = d_l/(1+z)
    term2 = (1+z) * c/H0 * 1/E_z
    ret = xp.abs(term1 + term2)
    return ret # H0 units, so MPc, km, s

def dz_dDl_analytic(d_l, z, args):
    ret = 1/dDl_dz_analytic(d_l, z, args)
    return ret

def dVc_dz_analytic_no_dl(z, args):
    H0, Om0, w = args
    c = 299792458 / 1000  # km/s
    d_l = d_l_func_jax(z, args)
    E_z = get_E_z(z, args)
    ret = (4 * xp.pi * (d_l**2) * c)/(H0 * E_z * (1+z)**2)
    return  ret


def dVc_dz_analytic(d_l, z, args):
    H0, Om0, w = args
    c = 299792458 / 1000  # km/s
    E_z = get_E_z(z, args)
    ret = (4 * xp.pi * (d_l**2) * c)/(H0 * E_z * (1+z)**2)
    return ret

if __name__ == "__main__":
    zmin = 0.001
    zmax = 2
    H0 = 67.66
    Om0 = 0.3111
    w = -1
    args = (H0, Om0, w)

    z = xp.linspace(1e-3, 2, 10000)
    d_l = d_l_func_jax(z, args)
    z_new = redshift_func_jax(d_l, zmin, zmax, args)
    ret = (z - z_new) / z

    assert xp.max(xp.abs(ret)) < 1e-6

    d_l = xp.array([1000, 40, 2000, 0.1])
    z = redshift_func_jax(d_l, zmin, zmax, args)
    d_l_new = d_l_func_jax(z, args)
    ret2 = (d_l - d_l_new) / d_l

    assert xp.max(xp.abs(ret2)) < 1e-6
    print("All tests passed")

