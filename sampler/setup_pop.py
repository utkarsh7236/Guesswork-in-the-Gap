import configparser
import os

import numpy as np

import preprocessing
import time
import shutil
import pickle
import importlib.util
import sys
import inspect
import pandas as pd
import cosmology as cosmo

config = configparser.ConfigParser()
config.read('config.ini')
# print(config.sections())

RUN_dir = config['DIRECTORIES']["run_dir"]
max_far = float(config["INJECTIONS"]["max_far"])

if not os.path.exists(RUN_dir):
    os.makedirs(RUN_dir)
    os.makedirs(RUN_dir+ "/data/")
    os.makedirs(RUN_dir+ "/results/")
    os.makedirs(RUN_dir + "/logs/")
    os.makedirs(RUN_dir + "/config/")
# else:
#     raise FileExistsError(f"{config['DIRECTORIES']['run_dir']} already exists")


CG_args = (float(config["COURSE_GRAIN"]["mass_lower"]), float(config["COURSE_GRAIN"]["mass_upper"]))
DIR_args = (config["DIRECTORIES"]["event_file_name"], config["DIRECTORIES"]["event_folder_name"],
            config["DIRECTORIES"]["vt_file_name"], config["DIRECTORIES"]["vt_folder_name"],
            config["DIRECTORIES"]["data_dir"])

with open(f"{RUN_dir}/data/wrangled.pkl", "wb") as f:
    data = preprocessing.load_data(CG_args, DIR_args, max_far)
    data_arg = preprocessing.wrangle(data)
    pickle.dump(data_arg, f)
    print("Data wrangling complete")

shutil.copy("config.ini", f"{RUN_dir}/config/config.ini")
shutil.copy("cosmology.py", f"{RUN_dir}/cosmology.py")
shutil.copy("cosmology.py", f"../cosmology.py")
shutil.copy("postprocessing_functions.py", f"{RUN_dir}/postprocessing_functions.py")
shutil.copy("preprocessing.py", f"{RUN_dir}/preprocessing.py")
shutil.copy(config["POPULATION"]["mass1d_func"], f"{RUN_dir}/config/mass1d_func.py")
shutil.copy(config["POPULATION"]["pairing_func"], f"{RUN_dir}/config/pairing_func.py")
shutil.copy(config["POPULATION"]["distance_func"], f"{RUN_dir}/config/distance_func.py")
shutil.copy(config["POPULATION"]["spin_func"], f"{RUN_dir}/config/spin_func.py")
shutil.copy("postprocessing.ipynb", f"{RUN_dir}/postprocessing.ipynb")


def _unravel(l):
    return ', '.join(str(i) for i in l)

def get_variables(file_path):
    module_name = file_path.split("/")[-1].split(".")[0]
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


mass1d_dict = get_variables(config["POPULATION"]["mass1d_func"])
pairing_dict = get_variables(config["POPULATION"]["pairing_func"])
distance_dict = get_variables(config["POPULATION"]["distance_func"])
spin_dict = get_variables(config["POPULATION"]["spin_func"])


mass1d_pop_vars = ["m"]
mass1d_lamda = [x for x in mass1d_dict[1] if x not in mass1d_pop_vars]

pairing_pop_vars = ["m1", "m2"]
pairing_lamda = [x for x in pairing_dict[1] if x not in pairing_pop_vars]

# mass_theta_vars = ["mass1_det", "mass2_det"]
mass_theta_vars = ["mass1_source", "mass2_source"]

distance_pop_vars = ["z"]
extra_distance_pop = ["mass1_source", "mass2_source"] # These are not "theta" params but show up in the spin function call for spin pops and need to be deleted
distance_lamda = [x for x in distance_dict[1] if x not in distance_pop_vars]
distance_lamda = [x for x in distance_lamda if x not in extra_distance_pop]

spin_pop_vars = ["a1", "costilt1", "a2", "costilt2"]
extra_spin_pop = ["mass1_source", "mass2_source"] # These are not "theta" params but show up in the spin function call for spin pops and need to be deleted
spin_lamda = [x for x in spin_dict[1] if x not in spin_pop_vars]
spin_lamda = [x for x in spin_lamda if x not in extra_spin_pop]

# Sensitive to the order of the variables, must match preprocessing.py
theta_vars = mass_theta_vars + distance_pop_vars + spin_pop_vars
lambda_vars = mass1d_lamda + pairing_lamda + distance_lamda + spin_lamda
lambda_vars = list(dict.fromkeys(lambda_vars))

with open(f"{RUN_dir}/lamda_ordered.txt", "w") as output:
    output.write(str(lambda_vars))

# print(mass1d_dict, pairing_dict, distance_dict, spin_dict, sep="\n"); print(lambda_vars); print(f"{','.join(str(i) for i in pairing_lamda)}") # THIS ONE PRINTS WITHOUT TUPLE; print(f"{*pairing_lamda,}") # THIS ONE PRINTS WITH TUPLE AND QUOTES AROUND EACH ELEMENT

# GENERATING model_vector.py FILE
with open(RUN_dir+"/model_vector.py", 'w') as f:
    f.write(f"""\
import jax
import jax.numpy as xp
jax.config.update("jax_enable_x64", True)
from cosmology import *
from config.mass1d_func import *
from config.pairing_func import *
from config.distance_func import *
from config.spin_func import *

def ln_prob_m_det(theta, lamda):
    {_unravel(theta_vars)} = theta
    ({_unravel(lambda_vars)}) = lamda
    pairing_func = lambda m1, m2: {pairing_dict[0]}(m1, m2, {_unravel(pairing_lamda)})
    mass_prob_func = lambda m: {mass1d_dict[0]}(m,{_unravel(mass1d_lamda)})
    prob_mass1_source = mass_prob_func(mass1_source)
    prob_mass2_source = mass_prob_func(mass2_source)
    ln_prob_joint_mass_source = xp.log(prob_mass1_source) + xp.log(prob_mass2_source) + xp.log(pairing_func(mass1_source, mass2_source))
    ret = ln_prob_joint_mass_source 
    ret = xp.where(mass1_source < mass2_source, -xp.inf, ret)
    return ret

def ln_prob_distance(theta, lamda):
    {_unravel(theta_vars)} = theta
    ({_unravel(lambda_vars)}) = lamda
    distance_func = {distance_dict[0]}(z, mass1_source, mass2_source, {_unravel(distance_lamda)})
    ret = xp.log(distance_func)
    return ret

def ln_prob_spin(theta, lamda):
    {_unravel(theta_vars)} = theta
    ({_unravel(lambda_vars)}) = lamda
    spin_prob_func = {spin_dict[0]}(mass1_source, mass2_source, a1, costilt1, a2, costilt2, {_unravel(spin_lamda)})
    ret = xp.log(spin_prob_func)
    return ret

def ln_prob(theta, lamda):
    return ln_prob_m_det(theta, lamda) + ln_prob_distance(theta, lamda) + ln_prob_spin(theta, lamda)

# Define the model
def model_vector(theta, lamda):
    return ln_prob(theta, lamda)
""")
print("Model vector created")


pairing_prior = pd.read_csv(config["PRIORS"]["pairing_prior"])
mass1d_prior = pd.read_csv(config["PRIORS"]["mass1d_prior"])
distance_prior = pd.read_csv(config["PRIORS"]["distance_prior"])
spin_prior = pd.read_csv(config["PRIORS"]["spin_prior"])
cosmology_prior = pd.read_csv(config["PRIORS"]["cosmology_prior"])

def setup_dVc_dz_dense(cosmology_prior, zmin = 0.00001, zmax = 2):
    _var = lambda var: cosmology_prior.loc[cosmology_prior['variable'] == var, 'prior'].values[0]
    args = [_var("H0"), _var("Om0"), _var("w")]
    z = np.linspace(zmin, zmax, 10000)
    dVc_dz = cosmo.dVc_dz_analytic_no_dl_old(z, args)
    np.savetxt(f"{RUN_dir}/dVc_dz_dense.txt", np.array([z, dVc_dz]).T)

# setup_dVc_dz_dense(cosmology_prior)

# Sort priors according to population params
df_lst = [pairing_prior, mass1d_prior, distance_prior, spin_prior, cosmology_prior]
df_lst = [df for df in df_lst if not df.empty]

all_priors = pd.concat(df_lst, axis=0)
all_priors['variable_aligned'] = pd.Categorical(
    all_priors['variable'],
    categories=lambda_vars,
    ordered=True
)
all_priors = all_priors.sort_values('variable_aligned')
del all_priors['variable_aligned']

python_prior = []

variable_df = pd.DataFrame()

for row in all_priors.iterrows():
    if not row[1]["inits"] == row[1]["inits"]:
        # string = f"{row[1]['variable']} = n.deterministic('{row[1]['variable']}', {row[1]['prior']})"
        string = f"{row[1]['variable']} = {row[1]['prior']}"
    else:
        string = f"{row[1]['variable']} = n.sample('{row[1]['variable']}', numpyro_dist.{row[1]['prior']})"
        variable_df = pd.concat([variable_df, pd.DataFrame([row[1]])], axis=0, ignore_index=True)
    python_prior.append(string)

del variable_df["prior"]
del variable_df["label"]
python_guess_args = dict(zip(variable_df["variable"], variable_df["inits"]))

# GENERATING priors.py FILE
with open(RUN_dir + "/priors.py", 'w') as f:
    f.write(f"""\
import jax
import jax.numpy as xp
jax.config.update("jax_enable_x64", True)
import numpyro as n
from numpyro import distributions as numpyro_dist


def prior():
{chr(10).join([f'{"    "}{item}' for item in python_prior])}
    
    lamda = [{_unravel(lambda_vars)}]
    return lamda
    
    
def get_guess_args(num_chains):
    g = lambda guess: guess * xp.ones(num_chains)
    guess_args = {python_guess_args}
    for key in guess_args:
        guess_args[key] = g(guess_args[key])
    return guess_args
    
""")
print("Prior file created")

num_warmup = int(config["HMC_PARAMS"]["num_warmup"])
num_samples = int(config["HMC_PARAMS"]["num_samples"])
num_chains = int(config["HMC_PARAMS"]["num_chains"])


# GENERATING hbi.py FILE
with open(RUN_dir + "/run_inference.py", 'w') as f:
    f.write(f"""\
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
    mcmc = MCMC(kernel, num_warmup={num_warmup}, num_samples={num_samples}, num_chains={num_chains}, progress_bar=True)
    mcmc.run(rng_key)
    
    prior_samples = infer_samples(mcmc)
    write_samples(mcmc, path = "results/prior")
    
    rng_key = jax.random.PRNGKey(0)
    kernel = NUTS(posterior)
    mcmc = MCMC(kernel, num_warmup={num_warmup}, num_samples={num_samples}, num_chains={num_chains}, progress_bar=True)
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


""")
print("Hierarchical Bayesian Inference file created")
print("Setup complete\n\n")

print("Run the following command to begin population inference:")
print(f"cd {RUN_dir}")
print(f"python3 run_inference.py")

