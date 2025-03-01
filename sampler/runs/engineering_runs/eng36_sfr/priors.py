import jax
import jax.numpy as xp
jax.config.update("jax_enable_x64", True)
import numpyro as n
from numpyro import distributions as numpyro_dist


def prior():
    alpha_1 = n.sample('alpha_1', numpyro_dist.Uniform(-5, 5))
    mu_peak = n.sample('mu_peak', numpyro_dist.Uniform(20, 100))
    sigma_peak = n.sample('sigma_peak', numpyro_dist.Uniform(2, 30))
    mixture = n.sample('mixture', numpyro_dist.Uniform(0, 1))
    model_min = 1
    beta_1 = n.sample('beta_1', numpyro_dist.Uniform(-5, 5))
    beta_2 = n.sample('beta_2', numpyro_dist.Uniform(-5, 5))
    sep = 5
    H0 = 67.32
    Om0 = 0.3158
    w = -1.0
    kappa = 2.7
    alpha_chi = n.sample('alpha_chi', numpyro_dist.Uniform(1, 5))
    beta_chi = n.sample('beta_chi', numpyro_dist.Uniform(1, 5))
    mix_tilt = n.sample('mix_tilt', numpyro_dist.Uniform(0, 1))
    sig_tilt = n.sample('sig_tilt', numpyro_dist.Uniform(0.1, 4))
    z_min = 0.0001
    z_max = 2.0
    
    lamda = [alpha_1, mu_peak, sigma_peak, mixture, model_min, beta_1, beta_2, sep, H0, Om0, w, kappa, alpha_chi, beta_chi, mix_tilt, sig_tilt]
    return lamda
    
    
def get_guess_args(num_chains):
    g = lambda guess: guess * xp.ones(num_chains)
    guess_args = {'alpha_1': -3.28, 'mu_peak': 35.0, 'sigma_peak': 10.0, 'mixture': 0.1, 'beta_1': 0.41, 'beta_2': 4.83, 'alpha_chi': 2.0, 'beta_chi': 2.0, 'mix_tilt': 0.5, 'sig_tilt': 1.0}
    for key in guess_args:
        guess_args[key] = g(guess_args[key])
    return guess_args
    
