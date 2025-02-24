import jax.numpy as xp
def powerlaw(m, alpha_1, model_min):
    ret = m ** alpha_1
    C = (-1*alpha_1 - 1) * (model_min ** (-1*alpha_1 - 1))
    return C*ret

def peak(m, lamda):
    mu, sig = lamda
    return 1 / (sig * xp.sqrt(2 * xp.pi)) * xp.exp(-0.5 * ((m - mu) / sig) ** 2)

def powerlaw_peak(m, alpha_1, mu_peak, sigma_peak, mixture, model_min):
    pl = powerlaw(m, alpha_1, model_min)
    G = peak(m, [mu_peak, sigma_peak])
    return (1 - mixture) * pl + mixture * G
