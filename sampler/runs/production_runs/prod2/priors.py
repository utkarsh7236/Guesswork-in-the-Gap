import jax
import jax.numpy as xp
jax.config.update("jax_enable_x64", True)
import numpyro as n
from numpyro import distributions as numpyro_dist


def prior():
    m_break = n.sample('m_break', numpyro_dist.Uniform(2, 10))
    alpha_1 = n.sample('alpha_1', numpyro_dist.Uniform(-5, 5))
    alpha_2 = n.sample('alpha_2', numpyro_dist.Uniform(-5, 5))
    gamma_low = n.sample('gamma_low', numpyro_dist.Uniform(2, 4))
    eta_low = n.sample('eta_low', numpyro_dist.Uniform(0, 50))
    gamma_high = n.sample('gamma_high', numpyro_dist.Uniform(4, 8))
    eta_high = n.sample('eta_high', numpyro_dist.Uniform(0, 50))
    A = n.sample('A', numpyro_dist.Uniform(0, 1))
    m_min = n.sample('m_min', numpyro_dist.Uniform(0.5, 1.2))
    eta_min = n.sample('eta_min', numpyro_dist.Uniform(10, 50))
    m_max = n.sample('m_max', numpyro_dist.Uniform(35, 100))
    eta_max = n.sample('eta_max', numpyro_dist.Uniform(0, 10))
    beta_1 = n.sample('beta_1', numpyro_dist.Uniform(-5, 5))
    beta_2 = n.sample('beta_2', numpyro_dist.Uniform(-5, 5))
    sep = 5
    H0 = 67.32
    Om0 = 0.3158
    w = -1.0
    kappa = n.sample('kappa', numpyro_dist.Uniform(-4, 8))
    mu_chi1 = n.sample('mu_chi1', numpyro_dist.Uniform(0, 0.4))
    sig_chi1 = n.sample('sig_chi1', numpyro_dist.Uniform(0.05, 0.5))
    mix_tilt1 = n.sample('mix_tilt1', numpyro_dist.Uniform(0, 1))
    sig_tilt1 = n.sample('sig_tilt1', numpyro_dist.Uniform(0.1, 4))
    mu_chi2 = n.sample('mu_chi2', numpyro_dist.Uniform(0, 1))
    sig_chi2 = n.sample('sig_chi2', numpyro_dist.Uniform(0.05, 2))
    mix_tilt2 = n.sample('mix_tilt2', numpyro_dist.Uniform(0, 1))
    sig_tilt2 = n.sample('sig_tilt2', numpyro_dist.Uniform(0.1, 4))
    m_spin_break = 3
    a_min = 0
    a_max = 1
    costilt_max = 1
    costilt_min = -1
    a_max_NS = 0.4
    z_min = 0.0001
    z_max = 2.0
    
    lamda = [m_break, alpha_1, alpha_2, gamma_low, eta_low, gamma_high, eta_high, A, m_min, eta_min, m_max, eta_max, beta_1, beta_2, sep, H0, Om0, w, kappa, mu_chi1, sig_chi1, mix_tilt1, sig_tilt1, mu_chi2, sig_chi2, mix_tilt2, sig_tilt2, m_spin_break, a_min, a_max, costilt_max, costilt_min, a_max_NS]
    return lamda
    
    
def get_guess_args(num_chains):
    g = lambda guess: guess * xp.ones(num_chains)
    guess_args = {'m_break': 5.0, 'alpha_1': -3.28, 'alpha_2': -1.15, 'gamma_low': 2.4, 'eta_low': 10.0, 'gamma_high': 5.5, 'eta_high': 10.0, 'A': 0.9, 'm_min': 0.8, 'eta_min': 20.0, 'm_max': 70.0, 'eta_max': 3.0, 'beta_1': 0.41, 'beta_2': 4.83, 'kappa': 2.7, 'mu_chi1': 0.1, 'sig_chi1': 0.25, 'mix_tilt1': 0.5, 'sig_tilt1': 1.0, 'mu_chi2': 0.1, 'sig_chi2': 1.0, 'mix_tilt2': 0.5, 'sig_tilt2': 1.0}
    for key in guess_args:
        guess_args[key] = g(guess_args[key])
    return guess_args
    
