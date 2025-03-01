import jax.numpy as xp
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

def pdb(m, m_break, alpha_1, alpha_2, gamma_low, eta_low, gamma_high, eta_high, A, m_min, eta_min, m_max, eta_max):
    _h = h(m, m_min, eta_min)
    _n = n_term(m, gamma_low, gamma_high, eta_low, eta_high, A)
    _l = l(m, m_max, eta_max)
    bpl = broken_power(m, alpha_1, alpha_2, m_break, 1)
    ret = _h * _n * _l * bpl
    return ret