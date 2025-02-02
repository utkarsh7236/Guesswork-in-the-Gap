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

def default_gwtc3(mass1_source, mass2_source, a1, costilt1, a2, costilt2, alpha, beta, mixtilt, sigtilt):
    costilt_max = 1
    costilt_min = -1

    p_chi1 = beta_pdf(a1, alpha, beta)
    p_chi2 = beta_pdf(a2, alpha, beta)

    mix_temp1 = truncnorm_pdf(costilt1, mu = 0, sigma=sigtilt, a=costilt_min, b=costilt_max)
    mix_temp2 = uniform_pdf(costilt1, costilt_min, costilt_max)
    p_costilt1 = mixtilt * mix_temp1 + (1-mixtilt) * mix_temp2

    mix_temp1 = truncnorm_pdf(costilt2, mu = 0, sigma=sigtilt, a=costilt_min, b=costilt_max)
    mix_temp2 = uniform_pdf(costilt2, costilt_min, costilt_max)
    p_costilt2 = mixtilt * mix_temp1 + (1-mixtilt) * mix_temp2

    p_s1 = p_chi1 * p_costilt1
    p_s2 = p_chi2 * p_costilt2
    return p_s1 * p_s2
