import configparser
import numpy
import numpy as np
import pandas as pd
from tqdm import tqdm

def compute_CG_frac(CG_args, pe): # Course grain by mass 1 source. Inside CG
    CG_LOWER, CG_UPPER = CG_args
    idx_inside_CG = pe.mass1_source[(pe.mass1_source > CG_LOWER) & (pe.mass1_source < CG_UPPER)].index
    # idx_inside_CG = pe.mass1_source[pe.mass1_source > CG_LOWER].index
    frac = len(idx_inside_CG)/len(pe.mass1_source)
    return frac

def load_data(CG_args, DIR_args, max_far, NUM_PE_SAMPLES = 4000):
    event_file_name, event_folder_name, vt_file_name, vt_folder_name, data_dir = DIR_args
    events = numpy.loadtxt(event_folder_name+event_file_name, dtype=str)
    NUM_PE_SAMPLES = int(NUM_PE_SAMPLES)
    CG_LOWER, CG_UPPER = CG_args

    columns_to_keep = ["mass1_source", "mass2_source",
                       "redshift", "a_1", "costilt1", "a_2", "costilt2",
                       "lnprob_mass1_source", "lnprob_mass2_source",
                       "lnprob_redshift",
                       "lnprob_spin1spherical", "lnprob_spin2spherical"]

    # Injections
    inj_deep = pd.DataFrame(numpy.genfromtxt(vt_folder_name + vt_file_name, delimiter=",", names=True))
    inj_deep = inj_deep[inj_deep["far"] < max_far]
    inj_deep = inj_deep.sample(frac=1, random_state=1)  # Do a random scramble to the dataset
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
    pe_shape = [NUM_PE_SAMPLES, len(columns_to_keep)]
    pe_full = numpy.zeros((*pe_shape, len_NCG))

    pe_names = np.array(columns_to_keep, dtype=str)
    inj_names = inj_deep.columns.to_numpy()
    CG_names = inj_deep.columns.to_numpy()

    # Not Course Grained PE
    print(f"Not in COURSE GRAIN: {len_NCG}")
    for i in tqdm(range(len(idx_NCG))):
        item = idx_NCG[i]
        pe = pd.DataFrame(numpy.genfromtxt(event_folder_name + events[item], delimiter=",", names=True))
        pe = pe.sample(n = NUM_PE_SAMPLES, random_state=1)
        for j in range(len(columns_to_keep)):
            pe_full[:, j, i] = pe[columns_to_keep[j]].to_numpy()

    # Course Grained INJ
    print(f"In COURSE GRAIN: {len_CG}")
    _pre_cut_CG = inj_deep.copy()
    # Treat everything OUTSIDE the CG as injections. The CG only knows the PE inside the CG, the rest is INJ
    # _post_cut_CG = _pre_cut_CG[(_pre_cut_CG["mass1_source"] < CG_LOWER) | (_pre_cut_CG["mass1_source"] > CG_UPPER)]
    _post_cut_CG = _pre_cut_CG[(_pre_cut_CG["mass1_source"] > CG_LOWER) & (_pre_cut_CG["mass1_source"] < CG_UPPER)]
    # _post_cut_CG = _pre_cut_CG[_pre_cut_CG["mass1_source"] < CG_LOWER]
    CG_full = _post_cut_CG.to_numpy()

    ret = [inj_deep.to_numpy(), pe_full, CG_full, inj_names, pe_names, CG_names, len_CG, len_NCG]
    return ret

def wrangle(data, split_data_arg = False, NUM_PE_SAMPLES = 4000):
    inj_deep, pe_full, CG_full, inj_names, pe_names, CG_names, N_CG, N_NCG = data
    NUM_PE_SAMPLES = int(NUM_PE_SAMPLES)

    # logZ - Not course grained
    cut = pe_full.shape[0]  # Total = 10,000
    _pe = lambda name: get_pe(name, pe_full[:cut, :, :], pe_names).squeeze()
    mass1_det = m_det(_pe("mass1_source"), _pe("redshift"))
    mass2_det = m_det(_pe("mass2_source"), _pe("redshift"))
    z = _pe("redshift")
    a1, costilt1 = _pe("a_1"), _pe("costilt1")
    a2, costilt2 = _pe("a_2"), _pe("costilt2")
    theta_pe = [_pe("mass1_source"), _pe("mass2_source"), z, a1, costilt1, a2, costilt2]

    importance_pe_lnprob_mass_src = _pe("lnprob_mass1_source") + _pe("lnprob_mass2_source") #- 2 * numpy.log((1 + _pe("redshift")))
    importance_pe_lnprob_spin1_spin2 = _pe("lnprob_spin1spherical") + _pe("lnprob_spin2spherical")
    importance_pe = importance_pe_lnprob_mass_src + _pe("lnprob_redshift") + importance_pe_lnprob_spin1_spin2

    len_NCG = len(_pe("redshift"))

    # log_eps
    injections = inj_deep.copy()
    _inj = lambda name: get_inj(name, injections, inj_names)
    mass1_det = m_det(_inj("mass1_source"), _inj("redshift"))
    mass2_det = m_det(_inj("mass2_source"), _inj("redshift"))
    z = _inj("redshift")
    a1, costilt1 = _inj("a_1"), _inj("costilt1")
    a2, costilt2 = _inj("a_2"), _inj("costilt2")
    # theta_inj = [mass1_det, mass2_det, z, a1, costilt1, a2, costilt2]
    theta_inj = [_inj("mass1_source"), _inj("mass2_source"), z, a1, costilt1, a2, costilt2]
    importance_inj = _inj("lnprob_mass1_source_mass2_source_redshift_spin1spherical_spin2spherical") #+ 2 * numpy.log(1 / (1 + _inj("redshift")))
    len_inj = len(_inj("redshift"))

    # logZ - Course grained
    _CG_full = CG_full.copy()  # Total = len(inj)
    _CG = lambda name: get_inj(name, _CG_full, CG_names)
    mass1_det = m_det(_CG("mass1_source"), _CG("redshift"))
    mass2_det = m_det(_CG("mass2_source"), _CG("redshift"))
    z = _CG("redshift")
    a1, costilt1 = _CG("a_1"), _CG("costilt1")
    a2, costilt2 = _CG("a_2"), _CG("costilt2")
    # theta_CG = [mass1_det, mass2_det, z, a1, costilt1, a2, costilt2]
    theta_CG = [_CG("mass1_source"), _CG("mass2_source"), z, a1, costilt1, a2, costilt2]
    importance_CG = _CG("lnprob_mass1_source_mass2_source_redshift_spin1spherical_spin2spherical") #+ 2 * numpy.log(1 / (1 + _CG("redshift")))
    len_CG = len(_CG("redshift"))

    data_arg = [theta_pe, importance_pe, theta_CG, importance_CG, theta_inj, importance_inj, len_NCG, len_CG, len_inj,
                N_CG]

    if split_data_arg:
        np.random.seed(1)
        first_indices_pe = np.random.choice(NUM_PE_SAMPLES, size=int(NUM_PE_SAMPLES/2), replace=False)
        second_indices_pe = np.setdiff1d(np.arange(NUM_PE_SAMPLES), first_indices_pe)
        _c_pe = lambda pe_arr: (pe_arr[first_indices_pe, :], pe_arr[second_indices_pe, :])
        _c = lambda arr: (arr[first_indices_pe], arr[second_indices_pe])

        assert _c_pe(_pe("mass1_source"))[0].shape[0] == int(NUM_PE_SAMPLES/2)
        mass1_source1, mass1_source2 = _c_pe(_pe("mass1_source"))
        mass2_source1, mass2_source2 = _c_pe(_pe("mass2_source"))
        z1, z2 = _c_pe(_pe("redshift"))
        a1_1, a1_2 = _c_pe(_pe("a_1"))
        costilt1_1, costilt1_2 = _c_pe(_pe("costilt1"))
        a2_1, a2_2 = _c_pe(_pe("a_2"))
        costilt2_1, costilt2_2 = _c_pe(_pe("costilt2"))

        assert mass1_source1.shape[0] == int(NUM_PE_SAMPLES/2)

        theta_pe1 = [mass1_source1, mass2_source1, z1, a1_1, costilt1_1, a2_1, costilt2_1]
        theta_pe2 = [mass1_source2, mass2_source2, z2, a1_2, costilt1_2, a2_2, costilt2_2]

        importance_pe1, importance_pe2 = _c(importance_pe)

        NUM_INJECTIONS = len(_inj("redshift"))
        first_indices_inj = np.random.choice(NUM_INJECTIONS, size=int(NUM_INJECTIONS / 2), replace=False)
        second_indices_inj = np.setdiff1d(np.arange(NUM_INJECTIONS), first_indices_pe)
        _c_inj = lambda inj_arr: (inj_arr[first_indices_inj], inj_arr[second_indices_inj])

        mass1_source1, mass1_source2 = _c_inj(_inj("mass1_source"))
        mass2_source1, mass2_source2 = _c_inj(_inj("mass2_source"))
        z1, z2 = _c_inj(_inj("redshift"))
        a1_1, a1_2 = _c_inj(_inj("a_1"))
        costilt1_1, costilt1_2 = _c_inj(_inj("costilt1"))
        a2_1, a2_2 = _c_inj(_inj("a_2"))
        costilt2_1, costilt2_2 = _c_inj(_inj("costilt2"))

        assert mass1_source1.shape[0] == int(NUM_INJECTIONS/2)

        theta_inj1 = [mass1_source1, mass2_source1, z1, a1_1, costilt1_1, a2_1, costilt2_1]
        theta_inj2 = [mass1_source2, mass2_source2, z2, a1_2, costilt1_2, a2_2, costilt2_2]

        importance_inj1, importance_inj2 = _c_inj(importance_inj)


        first_indices_CG = np.random.choice(N_CG, size=int(N_CG / 2), replace=False)
        second_indices_CG = np.setdiff1d(np.arange(N_CG), first_indices_pe)
        _c_cg = lambda cg_arr: (cg_arr[first_indices_CG], cg_arr[second_indices_CG])

        mass1_source1, mass1_source2 = _c_cg(_CG("mass1_source"))
        mass2_source1, mass2_source2 = _c_cg(_CG("mass2_source"))
        z1, z2 = _c_cg(_CG("redshift"))
        a1_1, a1_2 = _c_cg(_CG("a_1"))
        costilt1_1, costilt1_2 = _c_cg(_CG("costilt1"))
        a2_1, a2_2 = _c_cg(_CG("a_2"))
        costilt2_1, costilt2_2 = _c_cg(_CG("costilt2"))

        assert mass1_source1.shape[0] == int(N_CG/2)

        theta_CG1 = [mass1_source1, mass2_source1, z1, a1_1, costilt1_1, a2_1, costilt2_1]
        theta_CG2 = [mass1_source2, mass2_source2, z2, a1_2, costilt1_2, a2_2, costilt2_2]

        importance_CG1, importance_CG2 = _c_cg(importance_CG)
        data_arg1 = [theta_pe1, importance_pe1, theta_CG1, importance_CG1, theta_inj1, importance_inj1, len_NCG/2, len_CG/2, len_inj/2, N_CG/2]
        data_arg2 = [theta_pe2, importance_pe2, theta_CG2, importance_CG2, theta_inj2, importance_inj2, len_NCG/2, len_CG/2, len_inj/2, N_CG/2]
        return data_arg1, data_arg2
    return data_arg

def get_pe(str_name, pe_full, pe_names):
    return pe_full[:,pe_names == str_name,:].squeeze()

def get_inj(str_name, inj_deep, inj_names):
    return inj_deep[:, inj_names == str_name].squeeze()

def m_det(m_source, redshift):
    return m_source * (1 + redshift)