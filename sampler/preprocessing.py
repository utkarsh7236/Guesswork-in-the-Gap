import numpy
import pandas as pd
from tqdm import tqdm

def compute_CG_frac(CG_args, pe): # Course grain by mass 1 source. Inside CG
    CG_LOWER, CG_UPPER = CG_args
    idx_inside_CG = pe.mass1_source[(pe.mass1_source > CG_LOWER) & (pe.mass1_source < CG_UPPER)].index
    # idx_inside_CG = pe.mass1_source[pe.mass1_source > CG_LOWER].index
    frac = len(idx_inside_CG)/len(pe.mass1_source)
    return frac

def load_data(CG_args, DIR_args):
    event_file_name, event_folder_name, vt_file_name, vt_folder_name, data_dir = DIR_args
    events = numpy.loadtxt(event_folder_name+event_file_name, dtype=str)

    CG_LOWER, CG_UPPER = CG_args

    # Injections
    inj_deep = pd.DataFrame(numpy.genfromtxt(vt_folder_name + vt_file_name, delimiter=",", names=True))
    inj_deep = inj_deep[inj_deep["far"] < 1]
    inj_deep = inj_deep.sample(frac=1, random_state=1)  # Do a random scramble to the dataset
    d_l_arr, z_arr, grad = numpy.loadtxt(vt_folder_name + f"dl_dz_H0_injections.csv", unpack=True, delimiter=",", skiprows=1)
    injections_gradient = [d_l_arr, z_arr, grad]

    len_NCG = 0
    len_CG = 0
    idx_CG = []
    idx_NCG = []
    for i in range(len(events)):
        pe = pd.DataFrame(numpy.genfromtxt(event_folder_name + events[i], delimiter=",", names=True))
        frac = compute_CG_frac(CG_args, pe)
        if frac > 0.99:
            len_CG += 1
            idx_CG.append(i)
        else:
            len_NCG += 1
            idx_NCG.append(i)

    # Non-Course Grained
    pe_shape = pd.DataFrame(numpy.genfromtxt(event_folder_name + events[0], delimiter=",", names=True)).shape
    pe_gradient_shape = numpy.array([numpy.loadtxt(event_folder_name + f"dl_dz_H0_pe/{events[0]}.csv", unpack=True, delimiter=",",
                                                   skiprows=1)]).squeeze().T.shape
    pe_full = numpy.zeros((*pe_shape, len_NCG))
    pe_full_gradient = numpy.zeros((*pe_gradient_shape, len_NCG))

    pe_names = pd.DataFrame(numpy.genfromtxt(event_folder_name + events[0], delimiter=",", names=True)).columns.to_numpy()
    inj_names = inj_deep.columns.to_numpy()
    CG_names = inj_deep.columns.to_numpy()

    # Not Course Grained PE
    print(f"Not in COURSE GRAIN: {len_NCG}")
    for i in tqdm(range(len(idx_NCG))):
        item = idx_NCG[i]
        pe = pd.DataFrame(numpy.genfromtxt(event_folder_name + events[item], delimiter=",", names=True))
        pe = pe.sample(frac=1, random_state=1)
        pe_full[:, :, i] = pe.to_numpy()
        d_l_arr, z_arr, grad = numpy.loadtxt(event_folder_name + f"dl_dz_H0_pe/{events[item]}.csv", unpack=True, delimiter=",",
                                             skiprows=1)
        pe_gradient = numpy.array([d_l_arr, z_arr, grad])
        pe_full_gradient[:, :, i] = pe_gradient.squeeze().T

    # Course Grained INJ
    print(f"In COURSE GRAIN: {len_CG}")
    _pre_cut_CG = inj_deep.copy()
    # Treat everything OUTSIDE the CG as injections. The CG only knows the PE inside the CG, the rest is INJ
    # _post_cut_CG = _pre_cut_CG[(_pre_cut_CG["mass1_source"] < CG_LOWER) | (_pre_cut_CG["mass1_source"] > CG_UPPER)]
    _post_cut_CG = _pre_cut_CG[(_pre_cut_CG["mass1_source"] > CG_LOWER) & (_pre_cut_CG["mass1_source"] < CG_UPPER)]
    # _post_cut_CG = _pre_cut_CG[_pre_cut_CG["mass1_source"] < CG_LOWER]
    CG_full = _post_cut_CG.to_numpy()
    CG_full_gradient = injections_gradient

    ret = [inj_deep.to_numpy(), pe_full, CG_full, injections_gradient, pe_full_gradient, CG_full_gradient, inj_names,
           pe_names, CG_names, len_CG, len_NCG]
    return ret

def wrangle(data):
    inj_deep, pe_full, CG_full, injections_gradient, pe_full_gradient, CG_full_gradient, inj_names, pe_names, CG_names, N_CG, N_NCG = data
    assert inj_deep.shape, len(injections_gradient[0])

    # logZ - Not course grained
    cut = pe_full.shape[0]  # Total = 10,000
    _pe = lambda name: get_pe(name, pe_full[:cut, :, :], pe_names).squeeze()
    d_l_arr, z_arr, grad = numpy.rollaxis(pe_full_gradient[:cut, :, :], 1)
    d_dl_dz = grad
    mass1_det = m_det(_pe("mass1_source"), _pe("redshift"))
    mass2_det = m_det(_pe("mass2_source"), _pe("redshift"))
    d_l = _pe("luminosity_distance")
    spin1x, spin1y, spin1z = _pe("spin1x"), _pe("spin1y"), _pe("spin1z")
    spin2x, spin2y, spin2z = _pe("spin2x"), _pe("spin2y"), _pe("spin2z")
    theta_pe = [mass1_det, mass2_det, d_l, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z]
    importance_pe = _pe("lnprob_mass1_source") + _pe("lnprob_mass2_source") + _pe("lnprob_redshift") + \
                    _pe("lnprob_spin1x_spin1y_spin1z") + _pe("lnprob_spin2x_spin2y_spin2z") + 2 * numpy.log(
        1 / (1 + _pe("redshift"))) - numpy.log(numpy.abs(d_dl_dz))

    len_NCG = len(_pe("luminosity_distance"))
    pe_params = []

    # log_eps
    injections = inj_deep.copy()
    _inj = lambda name: get_inj(name, injections, inj_names)
    d_l_arr, z_arr, grad = injections_gradient
    d_l_val = _inj("luminosity_distance")
    d_dl_dz = numpy.interp(d_l_val, d_l_arr, grad)
    mass1_det = m_det(_inj("mass1_source"), _inj("redshift"))
    mass2_det = m_det(_inj("mass2_source"), _inj("redshift"))
    d_l = _inj("luminosity_distance")
    spin1x, spin1y, spin1z = _inj("spin1x"), _inj("spin1y"), _inj("spin1z")
    spin2x, spin2y, spin2z = _inj("spin2x"), _inj("spin2y"), _inj("spin2z")
    theta_inj = [mass1_det, mass2_det, d_l, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z]
    importance_inj = _inj(
        "lnprob_mass1_source_mass2_source_redshift_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z") + 2 * numpy.log(
        1 / (1 + _inj("redshift"))) - numpy.log(numpy.abs(d_dl_dz))
    len_inj = len(_inj("luminosity_distance"))

    # logZ - Course grained
    _CG_full = CG_full.copy()  # Total = len(inj)
    _CG = lambda name: get_inj(name, _CG_full, CG_names)
    d_l_arr, z_arr, grad = CG_full_gradient
    d_l_val = _CG("luminosity_distance")
    d_dl_dz = numpy.interp(d_l_val, d_l_arr, grad)
    mass1_det = m_det(_CG("mass1_source"), _CG("redshift"))
    mass2_det = m_det(_CG("mass2_source"), _CG("redshift"))
    d_l = _CG("luminosity_distance")
    spin1x, spin1y, spin1z = _CG("spin1x"), _CG("spin1y"), _CG("spin1z")
    spin2x, spin2y, spin2z = _CG("spin2x"), _CG("spin2y"), _CG("spin2z")
    theta_CG = [mass1_det, mass2_det, d_l, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z]
    importance_CG = _CG(
        "lnprob_mass1_source_mass2_source_redshift_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z") + 2 * numpy.log(
        1 / (1 + _CG("redshift"))) - numpy.log(numpy.abs(d_dl_dz))
    len_CG = len(_CG("luminosity_distance"))

    data_arg = [theta_pe, importance_pe, theta_CG, importance_CG, theta_inj, importance_inj, len_NCG, len_CG, len_inj,
                N_CG]
    return data_arg

def get_pe(str_name, pe_full, pe_names):
    return pe_full[:,pe_names == str_name,:].squeeze()

def get_inj(str_name, inj_deep, inj_names):
    return inj_deep[:, inj_names == str_name].squeeze()

def m_det(m_source, redshift):
    return m_source * (1 + redshift)