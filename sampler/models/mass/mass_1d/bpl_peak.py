import jax.numpy as xp

def broken_power(m, alpha_1, alpha_2, m_break, amp_break):
    ret = xp.where(m < m_break, amp_break * (m / m_break) ** (alpha_1), 0)
    ret = xp.where(m >= m_break, amp_break * (m / m_break) ** (alpha_2), ret)
    return ret

def peak(m, lamda):
    mu, sig = lamda
    return 1 / (sig * xp.sqrt(2 * xp.pi)) * xp.exp(-0.5 * ((m - mu) / sig) ** 2)

def bpl_peak(m, alpha_1, alpha_2, m_break, mu_peak, sigma_peak, mixture):
    bpl = broken_power(m, alpha_1, alpha_2, m_break, 1)
    G = peak(m, [mu_peak, sigma_peak])
    return (1 - mixture) * bpl + mixture * G
