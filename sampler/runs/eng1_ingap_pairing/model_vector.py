import jax
import jax.numpy as xp
jax.config.update("jax_enable_x64", True)
from cosmology import *
from config.mass1d_func import *
from config.pairing_func import *
from config.distance_func import *
from config.spin_func import *

def ln_prob_m_det(theta, lamda):
    mass1_det, mass2_det, z, a1, costilt1, a2, costilt2 = theta
    (m_break, alpha_1, alpha_2, gamma_low, eta_low, gamma_high, eta_high, A, m_min, eta_min, m_max, eta_max, mu_peak1, sig_peak1, peak_constant1, mu_peak2, sig_peak2, peak_constant2, mu_peak_NS, sig_peak_NS, peak_constant_NS, model_min, model_max, beta_1, beta_2, beta_gap, sep, H0, Om0, w, kappa, mu_chi1, sig_chi1, mix_tilt1, sig_tilt1, mu_chi2, sig_chi2, mix_tilt2, sig_tilt2, m_spin_break) = lamda
    redshift = z
    pairing_func = lambda m1, m2: beta_split_3(m1, m2, beta_1, beta_2, beta_gap, sep)
    mass_prob_func = lambda m: pdb_with_NG(m,m_break, alpha_1, alpha_2, gamma_low, eta_low, gamma_high, eta_high, A, m_min, eta_min, m_max, eta_max, mu_peak1, sig_peak1, peak_constant1, mu_peak2, sig_peak2, peak_constant2, mu_peak_NS, sig_peak_NS, peak_constant_NS, model_min, model_max)
    mass1_source = m_source(mass1_det, redshift)
    mass2_source = m_source(mass2_det, redshift)
    prob_mass1_source = mass_prob_func(mass1_source)
    prob_mass2_source = mass_prob_func(mass2_source)
    ln_prob_joint_mass_source = xp.log(prob_mass1_source) + xp.log(prob_mass2_source) + xp.log(pairing_func(mass1_source, mass2_source))
    jacobian = 2 * xp.log(1 / (1 + redshift))
    ret = ln_prob_joint_mass_source + jacobian
    ret = xp.where(mass1_det < mass2_det, -xp.inf, ret)
    return ret

def ln_prob_distance(theta, lamda):
    mass1_det, mass2_det, z, a1, costilt1, a2, costilt2 = theta
    (m_break, alpha_1, alpha_2, gamma_low, eta_low, gamma_high, eta_high, A, m_min, eta_min, m_max, eta_max, mu_peak1, sig_peak1, peak_constant1, mu_peak2, sig_peak2, peak_constant2, mu_peak_NS, sig_peak_NS, peak_constant_NS, model_min, model_max, beta_1, beta_2, beta_gap, sep, H0, Om0, w, kappa, mu_chi1, sig_chi1, mix_tilt1, sig_tilt1, mu_chi2, sig_chi2, mix_tilt2, sig_tilt2, m_spin_break) = lamda
    distance_func = powerlaw_distance(z, H0, Om0, w, kappa)
    ret = xp.log(distance_func)
    return ret

def ln_prob_spin(theta, lamda):
    mass1_det, mass2_det, z, a1, costilt1, a2, costilt2 = theta
    (m_break, alpha_1, alpha_2, gamma_low, eta_low, gamma_high, eta_high, A, m_min, eta_min, m_max, eta_max, mu_peak1, sig_peak1, peak_constant1, mu_peak2, sig_peak2, peak_constant2, mu_peak_NS, sig_peak_NS, peak_constant_NS, model_min, model_max, beta_1, beta_2, beta_gap, sep, H0, Om0, w, kappa, mu_chi1, sig_chi1, mix_tilt1, sig_tilt1, mu_chi2, sig_chi2, mix_tilt2, sig_tilt2, m_spin_break) = lamda
    mass1_source = m_source(mass1_det, z)
    mass2_source = m_source(mass2_det, z)
    spin_prob_func = newo4_break(mass1_source, mass2_source, a1, costilt1, a2, costilt2, mu_chi1, sig_chi1, mix_tilt1, sig_tilt1, mu_chi2, sig_chi2, mix_tilt2, sig_tilt2, m_spin_break)
    ret = xp.log(spin_prob_func)
    return ret

def ln_prob(theta, lamda):
    return ln_prob_m_det(theta, lamda) + ln_prob_distance(theta, lamda) + ln_prob_spin(theta, lamda)

# Define the model
def model_vector(theta, lamda):
    return ln_prob(theta, lamda)
