import warnings
import jax.numpy as xp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


VT_FOLDER_NAME = 'vt/O3_stiched/'
VT_FILE_NAME = "sensitivity-estimate.csv.gz"
EVENT_FOLDER_NAME = 'O3_data/'
EVENT_FILE_NAME = "event-list.txt"

def get_dl_dz_H0(d_l_arr, z_arr):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grad = np.gradient(d_l_arr, z_arr)
    mask = np.isnan(grad)
    grad[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), grad[~mask])
    return grad

def get_dl_dz_H0_PE(d_l_arr, z_arr, d_l_vals_to_interpolate):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grad = np.gradient(d_l_arr, z_arr)
    mask = np.isnan(grad)
    grad[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), grad[~mask])
    ret = np.interp(d_l_vals_to_interpolate, d_l_arr, grad)
    return ret

def analytic(d_l, z, args):
    H0, Om0, w = args
    c = 299792458 / 1000  # km/s
    omega_M = Om0
    omega_L = 0.6888463055445441
    omega_R = 0.00149369445
    omega_k = 1 - omega_M - omega_L - omega_R
    E_z = xp.sqrt(
        omega_M * (1 + z) ** 3 + omega_R * (1 + z) ** 4 + omega_k * (1 + z) ** 2 + omega_L * (1 + z) ** (3 * (1 + w)))

    term1 = d_l/(1+z)
    term2 = (1+z) * c/H0 * 1/E_z
    ret = np.abs(term1 + term2)
    return ret # H0 units, so MPc, km, s

# Generate for injection prior
injections = pd.DataFrame(np.genfromtxt(VT_FOLDER_NAME+VT_FILE_NAME, delimiter = ",", names=True))
df = injections.sort_values(by="luminosity_distance")
grad = get_dl_dz_H0(np.array(df["luminosity_distance"]), np.array(df["redshift"]))
d_l_arr = df["luminosity_distance"]
z_arr = df["redshift"]
np.savetxt(VT_FOLDER_NAME+f"dl_dz_H0_injections.csv", np.array([d_l_arr, z_arr, grad]).T, header="d_l_arr, z_arr, grad", delimiter=",")
inj_d_l = d_l_arr.copy()
inj_z = z_arr.copy()

# Generate for PE prior
events = np.loadtxt(EVENT_FOLDER_NAME+EVENT_FILE_NAME, dtype = str)
for i in tqdm(range(len(events))):
    path = EVENT_FOLDER_NAME + events[i]
    pe = pd.DataFrame(np.genfromtxt(path, delimiter=",",names=True))
    df = pe.sort_values(by="luminosity_distance")
    grad = get_dl_dz_H0_PE(inj_d_l, inj_z, np.array(df["luminosity_distance"]))
    d_l_arr = df["luminosity_distance"]
    z_arr = df["redshift"]
    np.savetxt(f"dl_dz_H0_pe/{events[i]}.csv", np.array([d_l_arr, z_arr, grad]).T, header="d_l_arr, z_arr, grad", delimiter=",")


# Compare injections prior to analytic
if __name__ == "__main__":
    d_l_val = [1000, 100, 2000, 5, 10000]
    grad_interp = np.interp(d_l_val, d_l_arr, grad)
    plt.scatter(d_l_val, grad_interp)
    plt.loglog(d_l_arr, grad, label="Interpolated") # Should be a little different due to different H0.
    plt.loglog(d_l_arr, analytic(np.array(d_l_arr), np.array(z_arr), [67.66, 0.3111, -1]), label="Analytic", linestyle = "--")
    plt.xlabel("Luminosity Distance (Mpc)")
    plt.ylabel("Gradient of Luminosity Distance wrt Redshift")
    plt.legend()
    plt.show()