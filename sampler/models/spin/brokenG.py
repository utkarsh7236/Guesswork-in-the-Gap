from cosmology import *
from jax.scipy.special import erf

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
    return xp.where((x >= a) & (x <= b), pdf, 1e-100)

def uniform_pdf(x, a, b):
    """Compute the uniform PDF using JAX."""
    return 1 / (b - a)

def prob_chi_single(a, mu_chi, sig_chi, a_min, a_max):
    p_chi = truncnorm_pdf(a, mu_chi, sig_chi, a_min, a_max)
    return p_chi

def prob_costilt_single(costilt, mix_tilt, sig_tilt, costilt_max, costilt_min):
    mix_temp1 = truncnorm_pdf(costilt, mu = 1, sigma=sig_tilt, a=costilt_min, b=costilt_max)
    mix_temp2 = uniform_pdf(costilt, costilt_min, costilt_max)
    p_costilt = mix_tilt * mix_temp1 + (1-mix_tilt) * mix_temp2
    return p_costilt

def prob_chi(a, m, mu_chi1, sig_chi1, mu_chi2, sig_chi2, m_spin_break, a_min, a_max, a_max_NS):
    p_chi_below = prob_chi_single(a, mu_chi1, sig_chi1, a_min, a_max_NS)
    p_chi_above = prob_chi_single(a, mu_chi2, sig_chi2, a_min, a_max)
    p_chi = xp.where(m < m_spin_break, p_chi_below, p_chi_above)
    return p_chi

def prob_costilt(costilt, m, mix_tilt1, sig_tilt1, mix_tilt2, sig_tilt2, m_spin_break, costilt_max, costilt_min):
    p_costilt_below = prob_costilt_single(costilt, mix_tilt1, sig_tilt1, costilt_max, costilt_min)
    p_costilt_above = prob_costilt_single(costilt, mix_tilt2, sig_tilt2, costilt_max, costilt_min)
    p_costilt = xp.where(m < m_spin_break, p_costilt_below, p_costilt_above)
    return p_costilt


def prob_spin_component(m, a, costilt,
                        mu_chi1, sig_chi1, mix_tilt1, sig_tilt1,
                        mu_chi2, sig_chi2, mix_tilt2, sig_tilt2,
                        m_spin_break,
                        a_min, a_max, costilt_max, costilt_min,
                        a_max_NS):

    # Chi distribution
    p_chi = prob_chi(a, m, mu_chi1, sig_chi1, mu_chi2, sig_chi2, m_spin_break, a_min, a_max, a_max_NS)

    # Costilt distribution
    p_costilt = prob_costilt(costilt, m, mix_tilt1, sig_tilt1, mix_tilt2, sig_tilt2, m_spin_break, costilt_max, costilt_min)

    # # Combine the two
    return p_chi * p_costilt

def brokenG(mass1_source, mass2_source, a1, costilt1, a2, costilt2,
                  mu_chi1, sig_chi1, mix_tilt1, sig_tilt1,
                  mu_chi2, sig_chi2, mix_tilt2, sig_tilt2,
                  m_spin_break,
                a_min, a_max, costilt_max, costilt_min,
                a_max_NS):
    p_s1 = prob_spin_component(mass1_source, a1, costilt1, mu_chi1, sig_chi1, mix_tilt1, sig_tilt1, mu_chi2, sig_chi2, mix_tilt2, sig_tilt2, m_spin_break, a_min, a_max, costilt_max, costilt_min,a_max_NS)
    p_s2 = prob_spin_component(mass2_source, a2, costilt2, mu_chi1, sig_chi1, mix_tilt1, sig_tilt1, mu_chi2, sig_chi2, mix_tilt2, sig_tilt2, m_spin_break, a_min, a_max, costilt_max, costilt_min,a_max_NS)
    return p_s1 * p_s2
