import jax
import jax.numpy as xp
jax.config.update("jax_enable_x64", True)
from cosmology import *
from config.mass1d_func import *
from config.pairing_func import *
from config.distance_func import *
from config.spin_func import *

def ln_prob_m_det(theta, lamda):
    mass1_source, mass2_source, z, a1, costilt1, a2, costilt2 = theta
    (alpha_1, alpha_2, m_break, mu_peak, sigma_peak, mixture, beta_1, beta_2, sep, H0, Om0, w, kappa, mu_chi1, sig_chi1, mix_tilt1, sig_tilt1, mu_chi2, sig_chi2, mix_tilt2, sig_tilt2, m_spin_break, a_min, a_max, costilt_max, costilt_min, a_max_NS) = lamda
    pairing_func = lambda m1, m2: beta_split(m1, m2, beta_1, beta_2, sep)
    mass_prob_func = lambda m: bpl_peak(m,alpha_1, alpha_2, m_break, mu_peak, sigma_peak, mixture)
    prob_mass1_source = mass_prob_func(mass1_source)
    prob_mass2_source = mass_prob_func(mass2_source)
    ln_prob_joint_mass_source = xp.log(prob_mass1_source) + xp.log(prob_mass2_source) + xp.log(pairing_func(mass1_source, mass2_source))
    ret = ln_prob_joint_mass_source 
    ret = xp.where(mass1_source < mass2_source, -xp.inf, ret)
    return ret

def ln_prob_distance(theta, lamda):
    mass1_source, mass2_source, z, a1, costilt1, a2, costilt2 = theta
    (alpha_1, alpha_2, m_break, mu_peak, sigma_peak, mixture, beta_1, beta_2, sep, H0, Om0, w, kappa, mu_chi1, sig_chi1, mix_tilt1, sig_tilt1, mu_chi2, sig_chi2, mix_tilt2, sig_tilt2, m_spin_break, a_min, a_max, costilt_max, costilt_min, a_max_NS) = lamda
    distance_func = powerlaw_redshift(z, mass1_source, mass2_source, H0, Om0, w, kappa)
    ret = xp.log(distance_func)
    return ret

def ln_prob_spin(theta, lamda):
    mass1_source, mass2_source, z, a1, costilt1, a2, costilt2 = theta
    (alpha_1, alpha_2, m_break, mu_peak, sigma_peak, mixture, beta_1, beta_2, sep, H0, Om0, w, kappa, mu_chi1, sig_chi1, mix_tilt1, sig_tilt1, mu_chi2, sig_chi2, mix_tilt2, sig_tilt2, m_spin_break, a_min, a_max, costilt_max, costilt_min, a_max_NS) = lamda
    spin_prob_func = newo4_break(mass1_source, mass2_source, a1, costilt1, a2, costilt2, mu_chi1, sig_chi1, mix_tilt1, sig_tilt1, mu_chi2, sig_chi2, mix_tilt2, sig_tilt2, m_spin_break, a_min, a_max, costilt_max, costilt_min, a_max_NS)
    ret = xp.log(spin_prob_func)
    return ret

def ln_prob(theta, lamda):
    return ln_prob_m_det(theta, lamda) + ln_prob_distance(theta, lamda) + ln_prob_spin(theta, lamda)

# Define the model
def model_vector(theta, lamda):
    return ln_prob(theta, lamda)
