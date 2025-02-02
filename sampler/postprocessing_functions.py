from cosmology import *
import corner
import numpy as np
import matplotlib.pyplot as plt
import io
import sys
def plot_sample_traces():
    pass


def get_lambda():
    # return m_arr, lamda, _lamda, _all_lambda
    pass

def get_correlations():
    pass

def plot_corner_all():
    pass

def plot_corner(variable1, variable2):
    pass

def save_summary(mcmc, outfile = "results/print_summary.txt"):
    summary_io = io.StringIO()
    sys.stdout = summary_io
    mcmc.print_summary()
    sys.stdout = sys.__stdout__  # Reset stdout

    # Write to file
    with open(outfile, "w") as f:
        f.write(summary_io.getvalue())

def save_corner(mcmc, posterior_samples, outfile = "results/corner.png"):
    params = get_non_deterministic_params(mcmc)
    params_samples = np.array([posterior_samples.get(key) for key in params]).T
    corner.corner(params_samples, labels=params, color="dodgerblue")
    plt.savefig(outfile)
    plt.close()

def get_non_deterministic_params(mcmc):
    from operator import attrgetter
    sites = mcmc._states[mcmc._sample_field]
    if isinstance(sites, dict) and True:
        state_sample_field = attrgetter(mcmc._sample_field)(mcmc._last_state)
        if isinstance(state_sample_field, dict):
            sites = {
                k: v
                for k, v in mcmc._states[mcmc._sample_field].items()
                if k in state_sample_field
            }
    return list(sites.keys())

def plot_traces(mcmc):
    import arviz as az
    data = az.from_numpyro(mcmc)
    az.plot_trace(data, var_names=get_non_deterministic_params(mcmc))