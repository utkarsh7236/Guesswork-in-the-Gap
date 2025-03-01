import jax
import jax.numpy as xp
jax.config.update("jax_enable_x64", True)
import numpyro as n
from numpyro import distributions as numpyro_dist


def prior():
    alpha_1 = n.sample('alpha_1', numpyro_dist.Uniform(-5, 5))
    alpha_2 = n.sample('alpha_2', numpyro_dist.Uniform(-5, 5))
    m_break = n.sample('m_break', numpyro_dist.Uniform(2, 10))
    mu_peak = n.sample('mu_peak', numpyro_dist.Uniform(20, 100))
    sigma_peak = n.sample('sigma_peak', numpyro_dist.Uniform(2, 30))
    mixture = n.sample('mixture', numpyro_dist.Uniform(0, 1))
    beta_1 = n.sample('beta_1', numpyro_dist.Uniform(-5, 5))
    beta_2 = n.sample('beta_2', numpyro_dist.Uniform(-5, 5))
    sep = 5
    H0 = 67.32
    Om0 = 0.3158
    w = -1.0
    kappa = n.sample('kappa', numpyro_dist.Uniform(-4, 8))
    mu_chi1 = n.sample('mu_chi1', numpyro_dist.Uniform(0, 0.4))
    sig_chi1 = n.sample('sig_chi1', numpyro_dist.Uniform(0.05, 2))
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
    
    lamda = [alpha_1, alpha_2, m_break, mu_peak, sigma_peak, mixture, beta_1, beta_2, sep, H0, Om0, w, kappa, mu_chi1, sig_chi1, mix_tilt1, sig_tilt1, mu_chi2, sig_chi2, mix_tilt2, sig_tilt2, m_spin_break, a_min, a_max, costilt_max, costilt_min, a_max_NS]
    return lamda
    
    
def get_guess_args(num_chains):
    g = lambda guess: guess * xp.ones(num_chains)
    guess_args = {'alpha_1': -3.28, 'alpha_2': -1.15, 'm_break': 5.0, 'mu_peak': 35.0, 'sigma_peak': 40.0, 'mixture': 0.1, 'beta_1': 0.41, 'beta_2': 4.83, 'kappa': 2.7, 'mu_chi1': 0.1, 'sig_chi1': 2.0, 'mix_tilt1': 0.5, 'sig_tilt1': 1.0, 'mu_chi2': 0.1, 'sig_chi2': 2.0, 'mix_tilt2': 0.5, 'sig_tilt2': 1.0}
    for key in guess_args:
        guess_args[key] = g(guess_args[key])
    return guess_args
    
