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
    kappa = n.sample('kappa', numpyro_dist.Uniform(-2, 8))
    mu_chi = n.sample('mu_chi', numpyro_dist.Uniform(0, 0.4))
    sig_chi = n.sample('sig_chi', numpyro_dist.Uniform(0.1, 1))
    mix_tilt = n.sample('mix_tilt', numpyro_dist.Uniform(0, 1))
    sig_tilt = n.sample('sig_tilt', numpyro_dist.Uniform(0.1, 4))
    a_max = 0.4
    a_min = 0
    costilt_max = 1
    costilt_min = -1
    z_min = 0.0001
    z_max = 2.0
    
    lamda = [alpha_1, alpha_2, m_break, mu_peak, sigma_peak, mixture, beta_1, beta_2, sep, H0, Om0, w, kappa, mu_chi, sig_chi, mix_tilt, sig_tilt, a_max, a_min, costilt_max, costilt_min]
    return lamda
    
    
def get_guess_args(num_chains):
    g = lambda guess: guess * xp.ones(num_chains)
    guess_args = {'alpha_1': -3.28, 'alpha_2': -1.15, 'm_break': 5.0, 'mu_peak': 35.0, 'sigma_peak': 40.0, 'mixture': 0.1, 'beta_1': 0.41, 'beta_2': 4.83, 'kappa': 2.7, 'mu_chi': 0.01, 'sig_chi': 0.2, 'mix_tilt': 0.5, 'sig_tilt': 1.0}
    for key in guess_args:
        guess_args[key] = g(guess_args[key])
    return guess_args
    
