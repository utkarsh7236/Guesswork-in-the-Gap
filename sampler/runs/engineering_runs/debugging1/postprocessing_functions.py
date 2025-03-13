from statistics import quantiles

from cosmology import *
import corner
import numpy as np
import io
import importlib.util
import sys
import numpy
from priors import *
import configparser
from preprocessing import load_data, wrangle
import inspect
import re
from tqdm import tqdm
import copy
import jax.numpy as xp

import matplotlib.pyplot as plt, pandas as pd, matplotlib as mpl, random


def utkarshGrid(): plt.minorticks_on(); plt.grid(color='grey', which='minor', linestyle=":",
                                                 linewidth='0.1', ); plt.grid(color='black', which='major',
                                                                              linestyle=":",
                                                                              linewidth='0.1', ); return None


def utkarshGridAX(ax): ax.minorticks_on(); ax.grid(color='grey', which='minor', linestyle=":",
                                                   linewidth='0.1', ); ax.grid(color='black', which='major',
                                                                               linestyle=":",
                                                                               linewidth='0.1', ); return None


mpl.rcParams['legend.frameon'], mpl.rcParams['figure.autolayout'] = False, True,
colour = ["dodgerblue", "goldenrod", "crimson", "teal", "yellowgreen", "grey"]
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": ["Palatino"], })


def utkarshWrapper(): plt.legend();plt.utkarshGrid();plt.gca().tick_params(direction='in', which='both', right=True,
                                                                           top=True); plt.tight_layout(); return None


def utkarshWrapperAX(ax): utkarshGridAX(ax); ax.tick_params(direction='in', which='both', right=True,
                                                            top=True); return None;


plt.utkarshGrid = utkarshGrid;
plt.utkarshWrapper = utkarshWrapper


def colour_sample(n=2, col=colour, seed=7236): random.seed(seed); return random.sample(colour, n)


plt.rcParams["image.cmap"] = "Set2"  # I recommend, Set2, Dark2
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colour)  # Use your own OR plt.cm.Set2.colors


def plot_sample_traces():
    pass


def get_lambda():
    # return m_arr, lamda, _lamda, _all_lambda
    pass


def get_correlations():
    pass


def plot_corner_all():
    pass


def plot_corner(variable1, variable2, label1, label2):
    corner.hist2d(variable1, variable2, color="dodgerblue", levels=[0.50, 0.90])
    plt.savefig(f"results/corner_{variable1}_{variable2}.png")
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.utkarshWrapper()
    pass


def save_summary(mcmc, outfile="results/print_summary.txt"):
    summary_io = io.StringIO()
    sys.stdout = summary_io
    mcmc.print_summary()
    sys.stdout = sys.__stdout__  # Reset stdout

    # Write to file
    with open(outfile, "w") as f:
        f.write(summary_io.getvalue())


def save_corner(mcmc, posterior_samples, outfile="results/corner.png"):
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


def get_config(file_path="config/config.ini"):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config


def get_variables(file_path, module_name=None):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    methods_info = {}

    # Requires function to have the same name as python file being imported.
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) or inspect.ismethod(obj):
            if name != module_name:
                continue
            signature = inspect.signature(obj)
            parameters = list(signature.parameters.keys())
            methods_info[name] = parameters
    return module_name, methods_info[f"{module_name}"]


def get_params_of_function(func_type=None, function_name=None):
    single_event_pe = ["m", "m1", "m2", "mass1_source", "mass2_source", "z", "a1", "a2", "costilt1", "costilt2"]
    redundant_hyperparams = ['H0', 'Om0', 'w']
    remove_lst = single_event_pe + redundant_hyperparams
    if function_name is None:
        function_name = get_config(file_path="config/config.ini")["POPULATION"][func_type].split("/")[-1][:-3]
    params_lst = get_variables(file_path=f"config/{func_type}.py", module_name=function_name)[1]
    params = [param for param in params_lst if param not in remove_lst]
    return params


def plot_p_m(posterior_samples, function=None):
    params = get_params_of_function(func_type="mass1d_func")
    params_samples = {param: posterior_samples[param] for param in params if param in posterior_samples.keys()}
    m = np.logspace(0, 2, 500)
    p_m = function(m, **params_samples, model_min=1, model_max=100)
    p50 = np.median(p_m, axis=0)
    p95 = np.percentile(p_m, 95, axis=0)
    p05 = np.percentile(p_m, 5, axis=0)
    # plt.plot(m, p_m.T, alpha = 1/255, color="grey")
    plt.fill_between(m, p05, p95, color="dodgerblue", alpha=0.2, label="90\% CI")
    plt.plot(m, p50, color="dodgerblue")
    plt.xscale("log");
    plt.yscale("log")
    plt.xlabel("m [M$_\odot$]");
    plt.ylabel("p(m)")
    plt.utkarshWrapper()
    plt.savefig("results/p_m.png", bbox_inches='tight')
    return p_m


def plot_p_pairing(posterior_samples):
    params = get_params_of_function(func_type="pairing_func")
    params_samples = {param: posterior_samples[param] for param in params if param in posterior_samples.keys()}
    fig, ax = plt.subplots(nrows=len(params_samples.keys()))
    for i in range(len(params_samples.keys())):
        ax[i].hist(params_samples[list(params_samples.keys())[i]], bins=50, color=colour_sample()[0], alpha=0.5)
        ax[i].set_xlabel(f"{list(params_samples.keys())[i]}")
        utkarshGridAX(ax[i])
    plt.savefig("results/pairing.png", bbox_inches='tight')
    return None


def plot_p_chi(posterior_samples, function=None):
    plt.figure()
    params = get_params_of_function(func_type="spin_func", function_name="prob_chi")
    params_samples = {param: posterior_samples[param] for param in params if param in posterior_samples.keys()}
    a = np.linspace(0, 1, 1000)

    try:
        p_a = function(a, **params_samples, a_min=0, a_max=0.4)
        p50 = np.median(p_a, axis=0)
        p95 = np.percentile(p_a, 95, axis=0)
        p05 = np.percentile(p_a, 5, axis=0)
        # plt.plot(a, p_a.T, alpha = 1/255, color="grey")
        plt.fill_between(a, p05, p95, color="dodgerblue", alpha=0.2, label="90\% CI")
        plt.plot(a, p50, color="dodgerblue")
    except:
        m_below = 1
        m_above = 10
        msb = 3
        p_a_below = function(a, m=m_below, **params_samples, m_spin_break=msb, a_min=0, a_max=1, a_max_NS=0.4)
        p_a_above = function(a, m=m_above, **params_samples, m_spin_break=msb, a_min=0, a_max=1, a_max_NS=0.4)
        p50_below = np.median(p_a_below, axis=0)
        p95_below = np.percentile(p_a_below, 95, axis=0)
        p05_below = np.percentile(p_a_below, 5, axis=0)
        p50_above = np.median(p_a_above, axis=0)
        p95_above = np.percentile(p_a_above, 95, axis=0)
        p05_above = np.percentile(p_a_above, 5, axis=0)
        plt.fill_between(a, p05_below, p95_below, color="dodgerblue", alpha=0.2, label="90\% CI")
        plt.plot(a, p50_below, color="dodgerblue", label=f"Below {msb} M$_\odot$")
        plt.fill_between(a, p05_above, p95_above, color="darkblue", alpha=0.2)
        plt.plot(a, p50_above, color="darkblue", label=f"Above {msb} M$_\odot$")
    plt.utkarshWrapper()
    plt.xlabel(r"$a$")
    plt.ylabel(r"$p(a)$")
    plt.savefig("results/p_a.png", bbox_inches='tight')
    return None


def plot_p_costilt(posterior_samples, function=None):
    plt.figure()
    params = get_params_of_function(func_type="spin_func", function_name="prob_costilt")
    params_samples = {param: posterior_samples[param] for param in params if param in posterior_samples.keys()}
    costilt = np.linspace(-1, 1, 1000)
    try:
        p_costilt = function(costilt, **params_samples, costilt_max=1, costilt_min=-1)
        p50 = np.median(p_costilt, axis=0)
        p95 = np.percentile(p_costilt, 95, axis=0)
        p05 = np.percentile(p_costilt, 5, axis=0)
        # plt.plot(a, p_a.T, alpha = 1/255, color="grey")
        plt.fill_between(costilt, p05, p95, color="crimson", alpha=0.2, label="90\% CI")
        plt.plot(costilt, p50, color="crimson")
    except:
        m_below = 1
        m_above = 10
        msb = 3
        p_costilt_below = function(costilt, m=m_below, **params_samples, m_spin_break=msb, costilt_max=1,
                                   costilt_min=-1)
        p_costilt_above = function(costilt, m=m_above, **params_samples, m_spin_break=msb, costilt_max=1,
                                   costilt_min=-1)
        p50_below = np.median(p_costilt_below, axis=0)
        p95_below = np.percentile(p_costilt_below, 95, axis=0)
        p05_below = np.percentile(p_costilt_below, 5, axis=0)
        p50_above = np.median(p_costilt_above, axis=0)
        p95_above = np.percentile(p_costilt_above, 95, axis=0)
        p05_above = np.percentile(p_costilt_above, 5, axis=0)
        plt.fill_between(costilt, p05_below, p95_below, color="crimson", alpha=0.2, label="90\% CI")
        plt.plot(costilt, p50_below, color="crimson", label=f"Below {msb} M$_\odot$")
        plt.fill_between(costilt, p05_above, p95_above, color="goldenrod", alpha=0.2)
        plt.plot(costilt, p50_above, color="goldenrod", label=f"Above {msb} M$_\odot$")
    plt.utkarshWrapper()
    plt.xlabel(r"$\cos\theta$")
    plt.ylabel(r"$p(\cos\theta)$")
    plt.savefig("results/p_costilt.png", bbox_inches='tight')
    return None


def plot_p_z(posterior_samples, function=None, H0=67.32, Om0=0.3158, w=-1.0):
    params = get_params_of_function(func_type="distance_func")
    plt.figure()
    params_samples = {param: posterior_samples[param] for param in params if param in posterior_samples.keys()}
    z = np.linspace(0, 2, 1000)
    mass1_source, mass2_source = 0, 0
    p_z = function(z, mass1_source, mass2_source, **params_samples, H0=H0, Om0=Om0, w=w)
    p50 = np.median(p_z, axis=0)
    p95 = np.percentile(p_z, 95, axis=0)
    p05 = np.percentile(p_z, 5, axis=0)
    # plt.plot(a, p_a.T, alpha = 1/255, color="grey")
    plt.fill_between(z, p05, p95, color="yellowgreen", alpha=0.2, label="90\% CI")
    plt.plot(z, p50, color="yellowgreen")
    plt.utkarshWrapper()
    plt.tight_layout()
    plt.xlabel(r"z")
    plt.ylabel(r"$p(z)$")
    plt.savefig("results/p_z.png", bbox_inches='tight')
    return p_z


def curate_data():
    # Curating the data
    config = configparser.ConfigParser()
    config.read('config/config.ini')
    RUN_dir = config['DIRECTORIES']["run_dir"]
    max_far = float(config["INJECTIONS"]["max_far"])
    CG_args = (float(config["COURSE_GRAIN"]["mass_lower"]), float(config["COURSE_GRAIN"]["mass_upper"]))
    translate_dir = "../../"
    DIR_args = (config["DIRECTORIES"]["event_file_name"],
                translate_dir + config["DIRECTORIES"]["event_folder_name"],
                config["DIRECTORIES"]["vt_file_name"],
                translate_dir + config["DIRECTORIES"]["vt_folder_name"],
                translate_dir + config["DIRECTORIES"]["data_dir"])
    data = load_data(CG_args, DIR_args, max_far=max_far)
    data_arg = wrangle(data)
    return data, data_arg


def posterior_samples_to_complete_numpy(posterior_samples):
    # get all elements of the first item in the dictionary
    len_hyperposterior = len(posterior_samples[list(posterior_samples.keys())[0]])
    sorted_lamda = list(numpy.loadtxt("lamda_ordered.txt", dtype=str))

    code_lines, line_no = inspect.getsourcelines(prior)
    # Regular expression to match the static variable assignments (excluding function calls and lists)
    pattern = r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^#]+?)\s*(?:#.*)?$'
    # Initialize the dictionary to store static variables
    static_vars = {}
    # Iterate over the lines
    for line in code_lines:
        # Exclude lines with 'n.sample' (function calls) or lines containing 'lamda' (list assignments)
        if 'n.sample' not in line and 'lamda' not in line:
            match = re.match(pattern, line.strip())
            if match:
                var_name = match.group(1)
                var_value = match.group(2).strip()
                static_vars[var_name] = var_value

    # Convert the items in static_vars to NumPy arrays with size len(posterior_samples)
    for var_name in static_vars:
        static_vars[var_name] = numpy.full(len_hyperposterior, static_vars[var_name])
    posterior_samples.update(static_vars)

    # Reorder the dictionary based on sorted_lambda
    assert sorted_lamda == list(
        {key: posterior_samples[key] for key in sorted_lamda}.keys())  # make sure the ordering is correct
    lambda_pop = numpy.array([posterior_samples[key].flatten() for key in sorted_lamda], dtype=float)
    return lambda_pop


def single_event_likelihood_fixedpop(theta_pe, importance_pe, single_lamda, model_vector):
    # Parameter Estimation
    theta_pe = xp.array(theta_pe)  # (7 single event params, 4000 samples, 66 events)
    N_pe = theta_pe.shape[1]
    N_events = theta_pe.shape[2]
    num = model_vector(theta_pe, single_lamda)  # shape (4000, 66)
    dem = importance_pe
    term1 = xp.exp(num - dem)  # xp.exp(num)/xp.exp(dem) = xp.exp(num - dem)
    ret = 1 / N_pe * xp.sum(term1, axis=0)  # shape (N_events)
    assert ret.shape[0] == N_events  # Make sure the single event likelihood has the shape of the number of events
    return ret  # Should have shape (single_lamda,N_pe)


def selection_fixedpop(theta_inj, importance_inj, single_lamda, model_vector):
    # Injections
    theta_inj = xp.array(theta_inj)  # (7 single event params, 202835 samples)
    N_draw = theta_inj.shape[1]
    num = model_vector(theta_inj, single_lamda)
    dem = importance_inj
    term1 = xp.exp(num - dem)  # xp.exp(num)/xp.exp(dem) = xp.exp(num - dem)
    ret = 1 / N_draw * xp.sum(term1, axis=0)  # one number
    assert len(importance_inj) == N_draw
    return ret  # Should have shape (single_lamda), just be a 1d array with the length of posterior samples.


def single_event_likelihood_variance(theta_pe, importance_pe, single_lamda, model_vector, mu_like):
    # Parameter Estimation
    theta_pe = xp.array(theta_pe)  # (7 single event params, 4000 samples, 66 events)
    N_pe = theta_pe.shape[1]
    N_events = theta_pe.shape[2]

    mult = 1 / (N_pe - 1) * 1 / N_pe
    num = model_vector(theta_pe, single_lamda)  # shape (66)
    dem = importance_pe
    frac = np.exp(num - dem)
    ret = mult * xp.sum((frac - mu_like) ** 2, axis=0)
    assert ret.shape[0] == N_events
    return ret  # Should have shape (single_lamda,N_pe)


def selection_variance(theta_inj, importance_inj, single_lamda, model_vector, mu_selection):
    theta_inj = xp.array(theta_inj)  # (7 single event params, 202835 samples)
    N_draw = theta_inj.shape[1]
    assert len(importance_inj) == N_draw

    mult1 = 1 / (N_draw - 1) * 1 / N_draw
    num = model_vector(theta_inj, single_lamda)
    dem = importance_inj
    frac = np.exp(num - dem)
    ret = mult1 * xp.sum((frac - mu_selection) ** 2)
    assert len(importance_inj) == N_draw
    return ret  # Should have length of single_lamda


# https://arxiv.org/pdf/2204.00461, Equation 53, A9
# O4 Astrodist, Equation A3-A6
# https://arxiv.org/pdf/1904.10879, Equation 9
def loglike_variance(theta_pe, importance_pe, theta_inj, importance_inj, single_lamda, model_vector):
    # compute for each sample of Lambda, thus an additional field is added to posterior sampels
    # Parameters to get
    N_events = xp.array(theta_pe).shape[2]
    N_draw = xp.array(theta_inj).shape[1]
    N_det = N_events

    mu_selection = selection_fixedpop(theta_inj, importance_inj, single_lamda, model_vector)
    mu_like = single_event_likelihood_fixedpop(theta_pe, importance_pe, single_lamda, model_vector)
    sig2_selection = selection_variance(theta_inj, importance_inj, single_lamda, model_vector, mu_selection)
    sig2_like = single_event_likelihood_variance(theta_pe, importance_pe, single_lamda, model_vector, mu_like)

    assert N_events == len(sig2_like)

    ret = xp.sum(sig2_like / (mu_like ** 2)) + (N_events ** 2) * sig2_selection / (mu_selection ** 2)
    # term1 = xp.sum(sig2_like/(mu_like**2))
    # term2 = sig2_selection/(mu_selection**2)
    # ret = term1 + (N_det**2) * term2

    loglike_var = ret
    neff_selection = (mu_selection ** 2) / sig2_selection
    neff_events = (mu_like ** 2) / sig2_like
    loglike_mu = xp.sum(xp.log(mu_like.squeeze())) - N_events * xp.log(mu_selection.squeeze())
    return loglike_mu, loglike_var, neff_selection, neff_events  # Should have length of posterior samples draws


def add_postprocessing_effects(posterior_samples, model_vector):
    data, data_arg = curate_data()
    theta_pe, importance_pe, theta_CG, importance_CG, theta_inj, importance_inj, len_NCG, len_CG, len_inj, N_CG = data_arg
    lambda_pop = posterior_samples_to_complete_numpy(posterior_samples)

    N_events = xp.array(theta_pe).shape[2]
    N_det = N_events

    neff_selection = []
    neff_events = []
    loglike_var = []
    loglike_mu = []
    neff_like = []

    posterior_samples_copy = copy.deepcopy(posterior_samples)

    i = 0
    for single_lamda in tqdm(lambda_pop.T):
        i += 1
        # if i > 20:  # only compute first 50 for now
        #     continue
        ll_mu, llv, nef_s, nef_e = loglike_variance(theta_pe, importance_pe, theta_inj, importance_inj, single_lamda,
                                                    model_vector)

        neff_selection.append(nef_s); loglike_var.append(llv); loglike_mu.append(ll_mu); neff_events.append(nef_e)

        nef_l = (np.exp(ll_mu)**2)/np.exp(llv) # (np.exp(ll_mu) ** 2) / np.exp(llv)  = np.exp((ll_mu**2) /llv) = np.exp((2*ll_mu) - llv)
        neff_like.append(nef_l)


    neff_selection = np.array(neff_selection); neff_events = np.array(neff_events); loglike_var = np.array(loglike_var)
    loglike_mu = np.array(loglike_mu); neff_like = np.array(neff_like)

    posterior_samples_copy["loglike_var"] = loglike_var; posterior_samples_copy["neff_selection"] = neff_selection
    posterior_samples_copy["neff_events"] = np.array(neff_events); posterior_samples_copy["neff_events_total"] = np.sum(posterior_samples_copy["neff_events"], axis=1)

    assert len(posterior_samples_copy["neff_events_total"]) == len(posterior_samples_copy["neff_selection"])
    neff_like_additive_inverse = 1/np.array(posterior_samples_copy["neff_events_total"]).squeeze() + (N_det ** 2)/np.array(posterior_samples_copy[
        "neff_selection"]).squeeze()
    posterior_samples_copy["neff_like_additive"] = 1/neff_like_additive_inverse
    posterior_samples_copy["neff_like"] = neff_like
    return posterior_samples_copy


def plot_neff(neff, filename):
    import os
    if not os.path.exists("results/monte_carlo_uncertainty"):
        os.makedirs("results/monte_carlo_uncertainty")

    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].hist(neff, bins=50, density=True)
    ax[1].ecdf(neff)
    plt.xlabel(f"Effective number of samples: {filename}")
    utkarshWrapperAX(ax[0]);
    utkarshWrapperAX(ax[1])
    plt.savefig(f"results/monte_carlo_uncertainty/neff_{filename}.png")
