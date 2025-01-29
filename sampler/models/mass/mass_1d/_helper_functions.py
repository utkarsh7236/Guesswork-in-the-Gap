import jax.numpy as xp
def broken_power(m, alpha_1, alpha_2, m_break, amp_break):
    ret = xp.where(m < m_break, amp_break * (m / m_break) ** (alpha_1), 0)
    ret = xp.where(m >= m_break, amp_break * (m / m_break) ** (alpha_2), ret)
    return ret

def peak(m, lamda):
    mu, sig = lamda
    return 1 / (sig * xp.sqrt(2 * xp.pi)) * xp.exp(-0.5 * ((m - mu) / sig) ** 2)

def l(m, m0, eta):
    return (1 + (m / m0) ** eta) ** (-1)


def h(m, m0, eta):
    return 1 - l(m, m0, eta)


def n_term(m, gamma_low, gamma_high, eta_low, eta_high, A):
    term = A * h(m, gamma_low, eta_low) * l(m, gamma_high, eta_high)
    return 1 - term