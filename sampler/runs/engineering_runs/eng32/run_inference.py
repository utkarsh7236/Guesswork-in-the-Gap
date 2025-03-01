import jax
import jax.numpy as xp
import h5py
jax.config.update("jax_enable_x64", True)
import numpyro as n
from numpyro.infer import MCMC, NUTS
from model_vector import *
from priors import *
from postprocessing_functions import *
import pickle
import os
import sys
sys.path.append('../')

def posterior(data_arg):
    theta_pe, importance_pe, theta_CG, importance_CG, theta_inj, importance_inj, len_NCG, len_CG, len_inj, N_CG = data_arg
    lamda = prior()

    # logZ (Single Event Evidences)
    # logZ (outside CG)
    num = model_vector(theta_pe, lamda)
    dem = importance_pe
    frac = num - dem  # xp.exp(num)/xp.exp(dem) = xp.exp(num - dem)
    summ1 = xp.sum(xp.exp(frac), axis=0)
    Z_i_NCG = summ1 / len_NCG
    n.factor("logZ_NCG", xp.sum(xp.log(Z_i_NCG)))

    # logZ (inside CG)
    num = model_vector(theta_CG, lamda)
    dem = importance_CG
    frac = num - dem  # xp.exp(num)/xp.exp(dem) = xp.exp(num - dem)
    summ3 = xp.sum(xp.exp(frac))
    Z_i_CG = summ3 / len_CG
    n.factor("logZ_CG", N_CG * xp.log(Z_i_CG))

    N_NCG = len(Z_i_NCG)  # Should equal N_NCG from data wrangling
    N_det = N_NCG + N_CG  # N_CG comes from cut in data wrangling

    # log_eps (Selection Function)
    num = model_vector(theta_inj, lamda)
    dem = importance_inj
    frac = num - dem  # x.exp(num)/xp.exp(dem) = xp.exp(num - dem)
    summ2 = xp.sum(xp.exp(frac))
    eps = summ2 / len_inj
    n.factor("log_eps", - N_det * xp.log(eps))
    return None
    
    
def infer_samples(mcmc):
    posterior_samples = mcmc.get_samples()
    return posterior_samples
    
    
def write_samples(mcmc, path):
    with h5py.File(path, "w") as f:
        for key, value in mcmc.get_samples().items():
            f.create_dataset(key, data=value)
    pickle_write(mcmc, path)
    
    
def pickle_write(mcmc, path):
    import pickle
    object_pi = mcmc
    with open(path + '_mcmc.obj', 'wb') as f:
        pickle.dump(object_pi, f)
    return None


def pickle_read(path):
    import pickle
    with open(path + '_mcmc.obj', 'rb') as f:
        unpickler = pickle.Unpickler(f)
        object_pi2 = unpickler.load()
    return object_pi2
    
if __name__ == "__main__":
    data_arg = pickle.load(open("data/wrangled.pkl", "rb"))
    rng_key = jax.random.PRNGKey(0)
    kernel = NUTS(prior)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=500, num_chains=1, progress_bar=True)
    mcmc.run(rng_key)
    
    prior_samples = infer_samples(mcmc)
    write_samples(mcmc, path = "results/prior")
    
    rng_key = jax.random.PRNGKey(0)
    kernel = NUTS(posterior)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=500, num_chains=1, progress_bar=True)
    mcmc.run(rng_key, data_arg, init_params=get_guess_args(1))
    
    posterior_samples = infer_samples(mcmc)
    write_samples(mcmc, path = "results/posterior")
    
    os.remove("data/wrangled.pkl")
    
    # Simple postprocessing
    try: 
        save_summary(mcmc, outfile = "results/print_summary.txt")
        save_corner(mcmc, posterior_samples, outfile="results/corner.png")
    except:
        print("Postprocessing failed")


