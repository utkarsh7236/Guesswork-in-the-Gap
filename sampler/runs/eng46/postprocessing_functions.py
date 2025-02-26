from cosmology import *
import corner
import numpy as np
import io
import importlib.util
import inspect
import configparser
import sys
import numpy
import matplotlib.pyplot as plt, pandas as pd, matplotlib as mpl, random
def utkarshGrid(): plt.minorticks_on() ; plt.grid(color='grey',which='minor',linestyle=":",linewidth='0.1',) ; plt.grid(color='black',which='major',linestyle=":",linewidth='0.1',); return None
def utkarshGridAX(ax): ax.minorticks_on() ; ax.grid(color='grey',which='minor',linestyle=":",linewidth='0.1',) ; ax.grid(color='black',which='major',linestyle=":",linewidth='0.1',); return None
mpl.rcParams['legend.frameon'], mpl.rcParams['figure.autolayout'] = False, True,
colour = ["dodgerblue", "goldenrod", "crimson", "teal", "yellowgreen", "grey"]
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": ["Palatino"],})
def utkarshWrapper(): plt.legend();plt.utkarshGrid() ;plt.gca().tick_params(direction='in', which='both', right=True, top=True); plt.tight_layout(); return None
def utkarshWrapperAX(ax): utkarshGridAX(ax); ax.tick_params(direction='in', which='both', right=True, top=True); return None;
plt.utkarshGrid = utkarshGrid; plt.utkarshWrapper = utkarshWrapper
def colour_sample(n = 2, col = colour, seed = 7236): random.seed(seed) ; return random.sample(colour, n)
plt.rcParams["image.cmap"] = "Set2" # I recommend, Set2, Dark2
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colour) # Use your own OR plt.cm.Set2.colors
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

def get_config(file_path = "config/config.ini"):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

def get_variables(file_path, module_name = None):
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

def get_params_of_function(func_type = None, function_name = None):
    single_event_pe = ["m", "m1", "m2", "mass1_source", "mass2_source", "z", "a1", "a2", "costilt1", "costilt2"]
    redundant_hyperparams = ['H0', 'Om0', 'w']
    remove_lst = single_event_pe + redundant_hyperparams
    if function_name is None:
        function_name = get_config(file_path = "config/config.ini")["POPULATION"][func_type].split("/")[-1][:-3]
    params_lst = get_variables(file_path=f"config/{func_type}.py", module_name=function_name)[1]
    params = [param for param in params_lst if param not in remove_lst]
    return params


def plot_p_m(posterior_samples, function = None):
    params = get_params_of_function(func_type = "mass1d_func")
    params_samples = {param: posterior_samples[param] for param in params if param in posterior_samples.keys()}
    m = np.logspace(0, 2, 500)
    p_m = function(m, **params_samples, model_min=1, model_max=100)
    p50 = np.median(p_m, axis=0)
    p95 = np.percentile(p_m, 95, axis=0)
    p05 = np.percentile(p_m, 5, axis=0)
    # plt.plot(m, p_m.T, alpha = 1/255, color="grey")
    plt.fill_between(m, p05, p95, color="dodgerblue", alpha = 0.2, label = "90\% CI")
    plt.plot(m, p50, color="dodgerblue")
    plt.xscale("log") ; plt.yscale("log")
    plt.xlabel("m [M$_\odot$]") ; plt.ylabel("p(m)")
    plt.utkarshWrapper()
    plt.savefig("results/p_m.png", bbox_inches='tight')
    return p_m

def plot_p_pairing(posterior_samples):
    params = get_params_of_function(func_type = "pairing_func")
    params_samples = {param: posterior_samples[param] for param in params if param in posterior_samples.keys()}
    fig, ax = plt.subplots(nrows=len(params_samples.keys()))
    for i in range(len(params_samples.keys())):
        ax[i].hist(params_samples[list(params_samples.keys())[i]], bins=50, color=colour_sample()[0], alpha=0.5)
        ax[i].set_xlabel(f"{list(params_samples.keys())[i]}")
        utkarshGridAX(ax[i])
    plt.savefig("results/pairing.png", bbox_inches='tight')
    return None

def plot_p_chi(posterior_samples, function  = None):
    plt.figure()
    params = get_params_of_function(func_type = "spin_func", function_name="prob_chi")
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
        p_a_below = function(a, m = m_below, **params_samples, m_spin_break = msb, a_min=0, a_max=1, a_max_NS=0.4)
        p_a_above = function(a, m = m_above, **params_samples, m_spin_break = msb, a_min=0, a_max=1, a_max_NS=0.4)
        p50_below = np.median(p_a_below, axis=0)
        p95_below = np.percentile(p_a_below, 95, axis=0)
        p05_below = np.percentile(p_a_below, 5, axis=0)
        p50_above = np.median(p_a_above, axis=0)
        p95_above = np.percentile(p_a_above, 95, axis=0)
        p05_above = np.percentile(p_a_above, 5, axis=0)
        plt.fill_between(a, p05_below, p95_below, color="dodgerblue", alpha=0.2, label="90\% CI")
        plt.plot(a, p50_below, color="dodgerblue", label = f"Below {msb} M$_\odot$")
        plt.fill_between(a, p05_above, p95_above, color="darkblue", alpha=0.2)
        plt.plot(a, p50_above, color="darkblue", label = f"Above {msb} M$_\odot$")
    plt.utkarshWrapper()
    plt.xlabel(r"$a$")
    plt.ylabel(r"$p(a)$")
    plt.savefig("results/p_a.png", bbox_inches='tight')
    return None

def plot_p_costilt(posterior_samples, function  = None):
    plt.figure()
    params = get_params_of_function(func_type = "spin_func", function_name="prob_costilt")
    params_samples = {param: posterior_samples[param] for param in params if param in posterior_samples.keys()}
    costilt = np.linspace(-1, 1, 1000)
    try:
        p_costilt = function(costilt, **params_samples, costilt_max=1, costilt_min=-1)
        p50 = np.median(p_costilt, axis=0)
        p95 = np.percentile(p_costilt, 95, axis=0)
        p05 = np.percentile(p_costilt, 5, axis=0)
        # plt.plot(a, p_a.T, alpha = 1/255, color="grey")
        plt.fill_between(costilt, p05, p95, color="crimson", alpha = 0.2, label = "90\% CI")
        plt.plot(costilt, p50, color="crimson")
    except:
        m_below = 1
        m_above = 10
        msb = 3
        p_costilt_below = function(costilt, m = m_below, **params_samples, m_spin_break = msb, costilt_max=1, costilt_min=-1)
        p_costilt_above = function(costilt, m = m_above, **params_samples, m_spin_break = msb, costilt_max=1, costilt_min=-1)
        p50_below = np.median(p_costilt_below, axis=0)
        p95_below = np.percentile(p_costilt_below, 95, axis=0)
        p05_below = np.percentile(p_costilt_below, 5, axis=0)
        p50_above = np.median(p_costilt_above, axis=0)
        p95_above = np.percentile(p_costilt_above, 95, axis=0)
        p05_above = np.percentile(p_costilt_above, 5, axis=0)
        plt.fill_between(costilt, p05_below, p95_below, color="crimson", alpha=0.2, label="90\% CI")
        plt.plot(costilt, p50_below, color="crimson", label = f"Below {msb} M$_\odot$")
        plt.fill_between(costilt, p05_above, p95_above, color="darkred", alpha=0.2)
        plt.plot(costilt, p50_above, color="darkred", label = f"Above {msb} M$_\odot$")
    plt.utkarshWrapper()
    plt.xlabel(r"$\cos\theta$")
    plt.ylabel(r"$p(\cos\theta)$")
    plt.savefig("results/p_costilt.png", bbox_inches='tight')
    return None

def plot_p_z(posterior_samples, function = None, H0 = 67.32, Om0 = 0.3158, w = -1.0):
    params = get_params_of_function(func_type = "distance_func")
    plt.figure()
    params_samples = {param: posterior_samples[param] for param in params if param in posterior_samples.keys()}
    z = np.linspace(0, 2, 1000)
    mass1_source, mass2_source = 0,0
    p_z = function(z, mass1_source, mass2_source, **params_samples, H0=H0, Om0=Om0, w=w)
    p50 = np.median(p_z, axis=0)
    p95 = np.percentile(p_z, 95, axis=0)
    p05 = np.percentile(p_z, 5, axis=0)
    # plt.plot(a, p_a.T, alpha = 1/255, color="grey")
    plt.fill_between(z, p05, p95, color="yellowgreen", alpha = 0.2, label = "90\% CI")
    plt.plot(z, p50, color="yellowgreen")
    plt.utkarshWrapper()
    plt.tight_layout()
    plt.xlabel(r"z")
    plt.ylabel(r"$p(z)$")
    plt.savefig("results/p_z.png", bbox_inches='tight')
    return p_z
