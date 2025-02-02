from cosmology import *
from jax.scipy.special import beta as beta_func
from jax.scipy.special import erf

def beta_pdf(x, a, b):
    """Compute Beta distribution PDF using JAX."""
    norm_const = 1 / beta_func(a, b)  # Normalization factor
    return norm_const * (x ** (a - 1)) * ((1 - x) ** (b - 1))

# Matches Colm Talbot from gwpopulation
# https://github.com/ColmTalbot/gwpopulation/blob/main/gwpopulation/utils.py
def truncnorm_pdf(x, mu, sigma, a, b):
    """Compute the analytically normalized truncated normal PDF using JAX."""
    # Compute normalization constant Z
    alpha = (a - mu) / (xp.sqrt(2) * sigma)
    beta = (b - mu) / (xp.sqrt(2) * sigma)
    Z = 0.5 * sigma * xp.sqrt(2 * xp.pi) * (erf(beta) - erf(alpha))
    # Compute the unnormalized Gaussian PDF
    pdf_unnormalized = xp.exp(-0.5 * ((x - mu) / sigma) ** 2)
    # Normalize
    pdf = pdf_unnormalized / Z
    return xp.where((x >= a) & (x <= b), pdf, 0.0)

def uniform_pdf(x, a, b):
    """Compute the uniform PDF using JAX."""
    return 1 / (b - a)

def prob_chi(a, alpha_chi, beta_chi):
    p_chi = beta_pdf(a, alpha_chi, beta_chi)
    return p_chi

def prob_costilt(costilt, mixtilt, sigtilt, costilt_max = 1, costilt_min = -1):
    mix_temp1 = truncnorm_pdf(costilt, mu = 0, sigma=sigtilt, a=costilt_min, b=costilt_max)
    mix_temp2 = uniform_pdf(costilt, costilt_min, costilt_max)
    p_costilt = mixtilt * mix_temp1 + (1-mixtilt) * mix_temp2
    return p_costilt

def prob_spin_component(mass1_source, mass2_source, a, costilt, alpha_chi, beta_chi, mixtilt, sigtilt):
    p_chi = prob_chi(a, alpha_chi, beta_chi)
    p_costilt = prob_costilt(costilt, mixtilt, sigtilt)
    return p_chi * p_costilt

def default_gwtc3(mass1_source, mass2_source, a1, costilt1, a2, costilt2, alpha_chi, beta_chi, mixtilt, sigtilt):
    p_s1 = prob_spin_component(mass1_source, mass2_source, a1, costilt1, alpha_chi, beta_chi, mixtilt, sigtilt)
    p_s2 = prob_spin_component(mass1_source, mass2_source, a2, costilt2, alpha_chi, beta_chi, mixtilt, sigtilt)
    return p_s1 * p_s2
