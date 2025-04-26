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
import shutil


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
    folder_path = "../../../sampler/runs/pdbNG_betaSplit_singleG_1_full/"
    priors_path = folder_path + "priors.py"
    posterior_samples_fixed = extract_equalities(priors_path)
    config_path = folder_path + "config/"
    mass_model_path = config_path + "mass1d_func.py"
    spin_model_path = config_path + "spin_func.py"
    conversion_dict = json.load(open("../../conversion_dictionaries/pdbNG_betaSplit_singleG.txt"))
    inv = {v: k for k, v in conversion_dict.items()}

    # Loading in population results
    posterior_path = folder_path + "results/posterior"
    with h5py.File(posterior_path, 'r') as f:
        posterior_samples = {key: np.array(f[key]) for key in f.keys()}

    merged_posterior_samples = merge_posterior_samples(posterior_samples, posterior_samples_fixed)

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

    skip_keys = ["switch_mass1_source_0", "switch_mass2_source_0"] # There is no switch point, since this is a nested model.

    for new_key, old_key in conversion_dict.items():
        print(f"[STATUS] Converting {old_key} to {new_key}")
        if new_key in skip_keys:
            converted_posterior_samples[new_key] = 1*np.ones((merged_posterior_samples["alpha_1"].shape[0], 1))
        elif new_key in gaussian_mixture_list:  # for items in the gaussian mixture list.
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

    # check for None values and print the keys
    for key, value in converted_posterior_samples.items():
        if value is None:
            print(f"Key '{key}' has a None value.")

    assert not any(value is None for value in converted_posterior_samples.values())

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