import json
import numpy as np
import re
import importlib
import inspect
from scipy.special import erf
import sys
import h5py
import jax.numpy as xp
import pickle
import csv
import gzip
import shutil
from scipy.signal import convolve
import importlib.util
import sys

# Set up argument parser
import argparse
parser = argparse.ArgumentParser(description="Process multiple population parameters")
parser.add_argument('--pop_param', nargs='+', required=True, help='List of population parameter names')
parser.add_argument('--pop_value', nargs='+', required=True, help='List of corresponding parameter values')

args = parser.parse_args()

# Convert values to floats if possible
pop_params = args.pop_param
pop_values = []
for val in args.pop_value:
    try:
        pop_values.append(float(val))
    except ValueError:
        pop_values.append(val)  # keep as string if not a number

# Combine into a dictionary for convenience
pop_dict = dict(zip(pop_params, pop_values))

print(f"[STATUS] Changing population parameters and values for:\n {pop_dict}")

def extract_equalities(file_path):
    equalities = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            # Match only lines with variable assignments not inside function calls
            if "=" in line and not line.startswith("def") and "(" not in line and not line.startswith("lamda") and "guess_args" not in line:
                key_value = re.split(r'\s*=\s*', line, maxsplit=1)
                if len(key_value) == 2:
                    key, value = key_value
                    equalities[key.strip()] = value.strip()
    return equalities

def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    #m = n / arrays[0].size
    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
        #for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out

def pickle_read(path):
    import pickle
    with open(path + '_mcmc.obj', 'rb') as f:
        unpickler = pickle.Unpickler(f)
        object_pi2 = unpickler.load()
    return object_pi2

def get_variables(file_path, model_name=None):
    module_name = file_path.split("/")[-1].split(".")[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    methods_info = {}

    # Requires function to have the same name as python file being imported.
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) or inspect.ismethod(obj):
            if name != model_name:
                continue
            signature = inspect.signature(obj)
            parameters = list(signature.parameters.keys())
            methods_info[name] = parameters
    return module_name, methods_info[f"{model_name}"]

def merge_posterior_samples(posterior_samples, posterior_samples_fixed):
    num_samples = next(iter(posterior_samples.values())).shape[0]  # Get the number of samples
    merged_dict = posterior_samples.copy()  # Start with the posterior samples

    for key, value in posterior_samples_fixed.items():
        if key not in merged_dict:  # Only add if not in posterior_samples
            merged_dict[key] = np.full((num_samples, 1), float(value))  # Create an array of the fixed value

    return merged_dict

if __name__ == "__main__":
    folder_path = "../../sampler/runs/pdbNG_betaSplit3_brokenG_1_full/"
    priors_path = folder_path + "priors.py"
    posterior_samples_fixed = extract_equalities(priors_path)
    posterior_samples_fixed["Ncomp"] = str(2.0)
    posterior_samples_fixed["Or0"] = str(0.0)
    # Converting units to what gwdistributions can understand
    posterior_samples_fixed["H0"] = str(float(posterior_samples_fixed["H0"]) * (1000/3.086e22))
    posterior_samples_fixed["OL0"] = str(0.6911) # 0.6842 Doesnt work with any other cosmology
    posterior_samples_fixed["Om0"] = str(0.3089) # 0.3158 Doesnt work with any other cosmology
    posterior_samples_fixed["max_redshift"] = str(4.0) # Using gw230529 defaults
    posterior_samples_fixed["min_redshift"] = str(0.0)
    posterior_samples_fixed["model_min"] = str(1.0)
    posterior_samples_fixed["model_max"] = str(100.0)
    config_path = folder_path + "config/"
    mass_model_path = config_path + "mass1d_func.py"
    spin_model_path = config_path + "spin_func.py"
    conversion_dict = json.load(open("../conversion_dictionaries/pdbNG_betaSplit3_brokenG.txt"))
    inv = {v: k for k, v in conversion_dict.items()}

    # Loading in population results
    posterior_path = folder_path + "results/posterior"
    with h5py.File(posterior_path, 'r') as f:
        posterior_samples = {key: np.array(f[key]) for key in f.keys()}

    merged_posterior_samples = merge_posterior_samples(posterior_samples, posterior_samples_fixed)
    
    kernel1 = np.array([-1, -2,  0,  2,  1])
    kernel2 = np.array([1,  -2,  1])

    spec = importlib.util.spec_from_file_location("mass1d_func", mass_model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mass1d_func"] = module
    spec.loader.exec_module(module)
    p_m_model = getattr(module, "pdb_with_NG")
    print(f"[STATUS] Using mass model: {p_m_model.__name__}")

    req_lst = [
    "m_break", "alpha_1", "alpha_2",
    "gamma_low", "eta_low", "gamma_high", "eta_high",
    "A", "m_min", "eta_min", "m_max", "eta_max",
    "mu_peak1", "sig_peak1", "peak_constant1",
    "mu_peak2", "sig_peak2", "peak_constant2",
    "mu_peak_NS", "sig_peak_NS", "peak_constant_NS",
    "model_min", "model_max"
    ]
    mass_grid = np.linspace(1, 100, 1000)  

    # Store all assigned variables in a dict
    param_values = {}

    for key in req_lst:
        if key not in merged_posterior_samples:
            raise ValueError(f"Missing required key: {key} in merged_posterior_samples")
        param_values[key] = merged_posterior_samples[key]  

    p_m_arr = np.zeros((len(param_values["m_break"]), len(mass_grid)))

    for i in range(len(param_values["m_break"])):
        param_list = [param_values[key][i] for key in req_lst]
        p_m_arr[i] = p_m_model(mass_grid, *param_list)

    # Make sure nothing in p_m_arr is NaN or Inf
    if np.any(np.isnan(p_m_arr)) or np.any(np.isinf(p_m_arr)):
        raise ValueError("p_m_arr contains NaN or Inf values. Check the mass model function.")

    m_tov_lst = []

    for p_m in p_m_arr:
        mask_full = (mass_grid >= 1) & (mass_grid <= 10)
        mass_roi = mass_grid[mask_full]
        p_m_roi = p_m[mask_full]
        filtered = convolve(np.log10(p_m_roi), kernel1, mode='same')
        search_mask = (mass_roi >= 1.8) & (mass_roi <= 5)
        filtered_search = filtered[search_mask]
        mass_search = mass_roi[search_mask]
        edge_idx_local = np.argmax(np.abs(filtered_search))
        mtov = mass_search[edge_idx_local]

        if mtov > 2 and mtov < 4.9:
            m_tov_lst.append(mtov)
            continue 
        else:
            filtered = convolve(np.log10(p_m_roi), kernel2, mode='same')
            search_mask = (mass_roi >= 1.8) & (mass_roi <= 5)
            filtered_search = filtered[search_mask]
            mass_search = mass_roi[search_mask]
            edge_idx_local = np.argmax(np.abs(filtered_search))
            mtov = mass_search[edge_idx_local]
            m_tov_lst.append(mtov)

    m_tov_arr = np.array(m_tov_lst)

    replace = lambda name, val: val*np.ones(merged_posterior_samples[name].shape)

    for key, val in pop_dict.items():
        print(f"[UPDATE] Replacing {key} with {val} in merged posterior samples")
        merged_posterior_samples[key] = replace(key, val)

    converted_posterior_samples = {}

    mu_tilt_list = ["mean_spin2_cos_polar_angle_spin2_polar_angle_1_mass2_source_0",
                    "mean_spin2_cos_polar_angle_spin2_polar_angle_1_mass2_source_1",
                    "mean_spin1_cos_polar_angle_spin1_polar_angle_1_mass1_source_1",
                    "mean_spin1_cos_polar_angle_spin1_polar_angle_1_mass1_source_0"]

    gaussian_mixture_list = ["sumgaussianpeak_prefactor_mass1_source_0", "sumgaussianpeak_prefactor_mass1_source_1",
                             "sumgaussianpeak_prefactor_mass1_source_2"]
    gaussian_mix_to_mu_sig_dict = {"sumgaussianpeak_prefactor_mass1_source_0": ["mu_peak1", "sig_peak1"],
                                   "sumgaussianpeak_prefactor_mass1_source_1": ["mu_peak2", "sig_peak2"],
                                   "sumgaussianpeak_prefactor_mass1_source_2": ["mu_peak_NS", "sig_peak_NS"]}

    gaussian_spin_list = ["mixture_frac_spin1_polar_angle_0_mass1_source_0",
                          "mixture_frac_spin1_polar_angle_1_mass1_source_0",
                          "mixture_frac_spin1_polar_angle_0_mass1_source_1",
                          "mixture_frac_spin1_polar_angle_1_mass1_source_1",
                          "mixture_frac_spin2_polar_angle_0_mass2_source_0",
                          "mixture_frac_spin2_polar_angle_1_mass2_source_0",
                          "mixture_frac_spin2_polar_angle_0_mass2_source_1",
                          "mixture_frac_spin2_polar_angle_1_mass2_source_1"]

    assert len(gaussian_spin_list) == len(set(gaussian_spin_list)), "Duplicate elements found in gaussian_spin_list"

    spin_mixture_duplicates = ["mixture_frac_spin1_polar_angle_1_mass1_source_0",
                               "mixture_frac_spin1_polar_angle_1_mass1_source_1",
                               "mixture_frac_spin2_polar_angle_1_mass2_source_0",
                               "mixture_frac_spin2_polar_angle_1_mass2_source_1"]
    gaussian_spin_to_mu_sig_dict = {
        "mixture_frac_spin1_polar_angle_1_mass1_source_0": "mixture_frac_spin1_polar_angle_0_mass1_source_0",
        "mixture_frac_spin1_polar_angle_1_mass1_source_1": "mixture_frac_spin1_polar_angle_0_mass1_source_1",
        "mixture_frac_spin2_polar_angle_1_mass2_source_0": "mixture_frac_spin2_polar_angle_0_mass2_source_0",
        "mixture_frac_spin2_polar_angle_1_mass2_source_1": "mixture_frac_spin2_polar_angle_0_mass2_source_1"}

    for new_key, old_key in conversion_dict.items():
        print(f"[STATUS] Converting {old_key} to {new_key}")
        if new_key in gaussian_mixture_list:  # for items in the gaussian mixture list.
            print("   Applying normalization offset to this conversion")
            converted_mu, converted_sigma = gaussian_mix_to_mu_sig_dict[new_key]
            mu, sigma = np.array(merged_posterior_samples[converted_mu]), np.array(
                merged_posterior_samples[converted_sigma])
            a = float(posterior_samples_fixed["model_min"])
            b = float(posterior_samples_fixed["model_max"])
            # normalization_offset = 1/(np.sqrt(2*np.pi)*sigma)
            alpha = (a - mu) / (xp.sqrt(2) * sigma)
            beta = (b - mu) / (xp.sqrt(2) * sigma)
            Z = 0.5 * sigma * xp.sqrt(2 * xp.pi) * (erf(beta) - erf(alpha))
            normalization_offset = 1 / Z
            converted_posterior_samples[new_key] = merged_posterior_samples[old_key] * normalization_offset
        elif new_key in spin_mixture_duplicates:  # for items in the gaussian spin list.
            print("   Mixture dictionary contains duplicates for this conversion, setting the right values")
            converted_posterior_samples[new_key] = merged_posterior_samples[old_key]
            converted_posterior_samples[gaussian_spin_to_mu_sig_dict[new_key]] = 1 - converted_posterior_samples[
                new_key]
        elif old_key in merged_posterior_samples:
            converted_posterior_samples[new_key] = merged_posterior_samples[old_key]
        elif new_key in mu_tilt_list:
            converted_posterior_samples[new_key] = np.ones((merged_posterior_samples["alpha_1"].shape[0], 1))
        else:
            converted_posterior_samples[new_key] = None  # or some placeholder if missing

    assert not any(value is None for value in converted_posterior_samples.values())

    conversion_dict["non_parametric_m_tov"] = "non_parametric_m_tov"
    inv["non_parametric_m_tov"] = "non_parametric_m_tov"
    converted_posterior_samples["non_parametric_m_tov"] = m_tov_arr.reshape(-1, 1)

    assert converted_posterior_samples["non_parametric_m_tov"].shape == converted_posterior_samples["notch_amplitude"].shape

    samples = []
    num_hyperparams = len(conversion_dict)
    num_samples = len(posterior_samples["alpha_1"])

    for i in range(num_samples):
        sample_dict = {}
        for key, value in converted_posterior_samples.items():
            sample_dict[key] = value[i].squeeze()
        samples.append(sample_dict)
    assert len(samples) == num_samples, f"Expected 5000 samples, but got {len(samples)}"
    assert len(samples[
                   0]) == num_hyperparams, f"Expected each dictionary to have {num_hyperparams} keys, but got {len(samples[0])}"

    # If no bugs, begin saving files
    # save paths as txt file
    with open("conversion_paths.txt", "w") as f:
        f.write(f"priors_path: {priors_path}\n")
        f.write(f"mass_model_path: {mass_model_path}\n")
        f.write(f"spin_model_path: {spin_model_path}\n")


    # save posterior_samples_fixed, conversion_dict and inv as dictionaries:
    with open("conversion_dict.json", "w") as f:
        json.dump(conversion_dict, f)
    with open("conversion_dict_inv.json", "w") as f:
        json.dump(inv, f)
    with open("posterior_samples_fixed.json", "w") as f:
        json.dump(posterior_samples_fixed, f)

    # save merged_posterior_samples as pkl file
    with open("merged_posterior_samples.pkl", "wb") as f:
        pickle.dump(merged_posterior_samples, f)

    with open("converted_posterior_samples.pkl", "wb") as f:
        pickle.dump(samples, f)

    # # copy mass_func1d and spin_func to directory
    # shutil.copy(mass_model_path, "mass_func1d.py")
    # shutil.copy(spin_model_path, "spin_func.py")

    # Write to gzip CSV
    with gzip.open(f"population{pop_values[0]}.csv.gz", "wt", newline="", encoding="utf-8") as gzfile:
        fieldnames = samples[0].keys()
        writer = csv.DictWriter(gzfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)