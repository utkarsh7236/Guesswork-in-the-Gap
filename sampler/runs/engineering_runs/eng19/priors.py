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
    kappa1 = n.sample('kappa1', numpyro_dist.Uniform(-2, 8))
    kappa2 = n.sample('kappa2', numpyro_dist.Uniform(-2, 8))
    m_break_kappa = 3
    z_min = 0.0001
    z_max = 2.0
    
    lamda = [alpha_1, alpha_2, m_break, mu_peak, sigma_peak, mixture, beta_1, beta_2, sep, H0, Om0, w, kappa1, kappa2, m_break_kappa]
    return lamda
    
    
def get_guess_args(num_chains):
    g = lambda guess: guess * xp.ones(num_chains)
    guess_args = {'alpha_1': -3.28, 'alpha_2': -1.15, 'm_break': 5.0, 'mu_peak': 35.0, 'sigma_peak': 40.0, 'mixture': 0.1, 'beta_1': 0.41, 'beta_2': 4.83, 'kappa1': 2.7, 'kappa2': 2.7}
    for key in guess_args:
        guess_args[key] = g(guess_args[key])
    return guess_args
    
