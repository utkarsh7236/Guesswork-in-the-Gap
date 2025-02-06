import jax.numpy as xp
from jax.scipy.special import erf

def broken_power(m, alpha_1, alpha_2, m_break, amp_break):
    ret = xp.where(m < m_break, amp_break * (m / m_break) ** (alpha_1), 0)
    ret = xp.where(m >= m_break, amp_break * (m / m_break) ** (alpha_2), ret)
    return ret

def l(m, m0, eta):
    return (1 + (m / m0) ** eta) ** (-1)

def h(m, m0, eta):
    return 1 - l(m, m0, eta)

def n_term(m, gamma_low, gamma_high, eta_low, eta_high, A):
    term = A * h(m, gamma_low, eta_low) * l(m, gamma_high, eta_high)
    return 1 - term

def pdb(m, m_break, alpha_1, alpha_2, gamma_low, eta_low, gamma_high, eta_high, A, m_min, eta_min, m_max, eta_max, model_min, model_max):
    _h = h(m, m_min, eta_min)
    _n = n_term(m, gamma_low, gamma_high, eta_low, eta_high, A)
    _l = l(m, m_max, eta_max)
    bpl = broken_power(m, alpha_1, alpha_2, m_break, 1)
    ret = _h * _n * _l * bpl

    # Apply model limits
    ret = xp.where((m <= model_max) & (m >= model_min), ret, 0.0) # Hopefully this is small enough
    return ret

# Matches Colm Talbot from gwpopulation
# https://github.com/ColmTalbot/gwpopulation/blob/main/gwpopulation/utils.py
def truncnorm(x, lamda):
    """Compute the analytically normalized truncated normal PDF using JAX."""
    # Compute normalization constant Z
    mu, sigma, a, b = lamda
    alpha = (a - mu) / (xp.sqrt(2) * sigma)
    beta = (b - mu) / (xp.sqrt(2) * sigma)
    Z = 0.5 * sigma * xp.sqrt(2 * xp.pi) * (erf(beta) - erf(alpha))
    # Compute the unnormalized Gaussian PDF
    pdf_unnormalized = xp.exp(-0.5 * ((x - mu) / sigma) ** 2)
    # Normalize
    pdf = pdf_unnormalized / Z
    return xp.where((x >= a) & (x <= b), pdf, 0.0)

def multi_pdb(m, m_break, alpha_1, alpha_2, gamma_low, eta_low, gamma_high, eta_high, A, m_min, eta_min, m_max, eta_max, mu_peak1, sig_peak1, peak_constant1, mu_peak2, sig_peak2, peak_constant2, model_min, model_max):
    m1 = pdb(m, m_break, alpha_1, alpha_2, gamma_low, eta_low, gamma_high, eta_high, A, m_min, eta_min, m_max, eta_max, model_min, model_max)
    m2 = peak_constant1 * truncnorm(m, [mu_peak1, sig_peak1, model_min, model_max])
    m3 = peak_constant2 * truncnorm(m, [mu_peak2, sig_peak2, model_min, model_max])
    return m1 * (1 + m2 + m3)
