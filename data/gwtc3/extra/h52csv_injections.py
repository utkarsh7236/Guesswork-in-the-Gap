#!/usr/bin/env python

"""a quick script to grab injection parameters and convert them into a standardized format
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import os

import h5py
import numpy as np

from argparse import ArgumentParser

### non-standard libraries
from conversion_cosmology import \
    (Cosmology, PLANCK_2018_Ho, PLANCK_2018_OmegaLambda, PLANCK_2018_OmegaMatter, PLANCK_2018_OmegaRadiation, MPC_CGS)

#-------------------------------------------------

names = {
    'mass1_source' : 'mass1_source',
    'mass2_source' : 'mass2_source',
    'redshift' : 'redshift',
    # 'distance' : 'luminosity_distance',
    'spin1x' : 'spin1x',
    'spin1y' : 'spin1y',
    'spin1z' : 'spin1z',
    'spin2x' : 'spin2x',
    'spin2y' : 'spin2y',
    'spin2z' : 'spin2z',
    # 'declination' : 'declination',
    # 'right_ascension' : 'right_ascension',
    # 'gps_time' : 'geocenter_time',
}

fars = {
    'far_cwb' : 'far_cwb',
    'far_gstlal' : 'far_gstlal',
    'far_mbta' : 'far_mbta',
    'far_pycbc_bbh' : 'far_pycbc_bbh',
    'far_pycbc_hyperbank' : 'far_pycbc_hyperbank',
}

log = {
    # 'declination_sampling_pdf' : 'lnprob_declination',
    # 'right_ascension_sampling_pdf' : 'lnprob_right_ascension',
    'sampling_pdf' : 'lnprob_mass1_source_mass2_source_redshift_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z',
}

#-------------------------------------------------

parser = ArgumentParser(description=__doc__)

parser.add_argument('source', type=str)
parser.add_argument('target', type=str)

parser.add_argument('--max-far', default=100, type=float,
    help='the maximum allowed far for an event to be considered detectable. \
As long as this is well above any reasonable detection threshold, it should not affect the inference. \
DEFAULT=100 [1/yr]')

parser.add_argument('--Ho', default=PLANCK_2018_Ho, type=float,
    help='Hubble parameter at z=0 in CGS. DEFAULT=%.6e'%PLANCK_2018_Ho)
parser.add_argument('--OmegaLambda', default=PLANCK_2018_OmegaLambda, type=float,
    help='DEFAULT=%.6f'%PLANCK_2018_OmegaLambda)
parser.add_argument('--OmegaMatter', default=PLANCK_2018_OmegaMatter, type=float,
    help='DEFAULT=%.6f'%PLANCK_2018_OmegaMatter)
parser.add_argument('--OmegaRadiation', default=PLANCK_2018_OmegaRadiation, type=float,
    help='DEFAULT=%.6e'%PLANCK_2018_OmegaRadiation)

parser.add_argument('-v', '--verbose', default=False, action='store_true')

args = parser.parse_args()

#-------------------------------------------------

if args.verbose:
    print('''instantiating cosmology:
    Ho=%.6e
    OmegaMatter=%.6e
    OmegaRadiation=%.6e
    OmegaLambda=%.6e'''%(args.Ho, args.OmegaMatter, args.OmegaRadiation, args.OmegaLambda))
cosmo = Cosmology(args.Ho, args.OmegaMatter, args.OmegaRadiation, args.OmegaLambda)

#-------------------------------------------------

if args.verbose:
    print('loading : '+args.source)

data = dict()
attrs = dict()
with h5py.File(args.source, 'r') as obj:

    nsmp = obj['injections'].attrs['n_accepted']
    attrs['analysis_time_s'] = obj['injections'].attrs['analysis_time_s']
    attrs['total_generated'] = obj['injections'].attrs['total_generated']

    # standard variables, including draw probabilties
    for old, new in list(names.items()) + list(fars.items()):
        if args.verbose:
            print('    injections/%s --> %s'%(old, new))
        data[new] = obj['injections'][old][:]

    # draw probabilities
    for old, new in log.items():
        if args.verbose:
            print('    log(injections/%s) --> %s'%(old, new))
        data[new] = np.log(obj['injections'][old][:])

#---

### compute the far for selection as the minimum of all searches
### remove individual search fars to avoid confusion
if args.verbose:
    print('computing far as the minimum over searches')
data['far'] = np.min([data.pop(val) for val in fars.values()], axis=0)

detectable = data['far'] < args.max_far ### only the events that could ever be detected
if args.verbose:
    print('retaining %d / %d detectable events (FAR < %.3e)'%(np.sum(detectable), len(detectable), args.max_far))
for key, val in data.items():
    data[key] = val[detectable]

#---

### convert parameters to what they would be with our cosmology
### update draw probabilities
if args.verbose:
    print('updating source-frame parameters to our cosmology')

### grab the injection parameters
mass1_source = data.pop('mass1_source')
mass2_source = data.pop('mass2_source')

redshift = data.pop('redshift')
luminosity_distance = cosmo.z2DL(redshift)
#luminosity_distance = data.pop('luminosity_distance') * MPC_CGS ### my cosmology works in CGS units, not Mpc
# luminosity_distance = data['luminosity_distance'] * MPC_CGS ### my cosmology works in CGS units, not Mpc

mass1_detector = mass1_source * (1.+redshift)
mass2_detector = mass2_source * (1.+redshift)

lnprob = data['lnprob_mass1_source_mass2_source_redshift_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z']

### extend cosmology to required redshift
cosmo.extend(max_DL=np.max(luminosity_distance))

### now compute the draw probabilities and parameters in the source frame given our new cosmology
data['redshift'] = cosmo.DL2z(luminosity_distance) ### the redshift in the new cosmology

# update source-frame parameters
data['mass1_source'] = mass1_detector / (1.+data['redshift']) ### source-frame masses in the new cosmology
data['mass2_source'] = mass2_detector / (1.+data['redshift'])

# update draw probability with jacobians
lnjac_mass1_source = lnjac_mass2_source = np.log(1.+redshift) - np.log(1.+data['redshift']) ### jacobian for masses

# from old redshift to luminosity_distance
# build an array for redshift, luminosity_distance in the old cosmology
# use this to numerically estimate the derivative at all the sample points
print('    WARNING: estimating jacobian(z->DL) for old cosmology with numeric derivative')

order = np.argsort(redshift) ### smallest to largest
redshift_list = [redshift[order[0]]]
luminosity_distance_list = [luminosity_distance[order[0]]]
for ind in order[1:]:
    ### require monotonicity and a nontrivial step size to avoid rounding errors
    if (redshift[ind] > redshift_list[-1]+0.001) and (luminosity_distance[ind] > luminosity_distance_list[-1]):
        redshift_list.append(redshift[ind])
        luminosity_distance_list.append(luminosity_distance[ind])
redshift_list = np.array(redshift_list)
luminosity_distance_list = np.array(luminosity_distance_list)

dDLdz_old = np.interp(redshift, redshift_list, np.gradient(luminosity_distance_list, redshift_list))

# from luminosity_distance to new redshift
dDLdz_new = cosmo.z2Dc(data['redshift']) + (1.+data['redshift'])*cosmo.dDcdz(data['redshift'])

# combine these to get old redshift to new redshift
lnjac_redshift = np.log(dDLdz_new) - np.log(dDLdz_old) - np.log(1+data['redshift'])

# conversions for spin
data["a_1"] = np.sqrt(data["spin1x"]**2 + data["spin1y"]**2 + data["spin1z"]**2)
data["a_2"] = np.sqrt(data["spin2x"]**2 + data["spin2y"]**2 + data["spin2z"]**2)
data["costilt1"] = data["spin1z"]/data["a_1"]
data["costilt2"] = data["spin2z"]/data["a_2"]
ln_jac_spin1 = 2 * np.log(data["a_1"])
ln_jac_spin2 = 2 * np.log(data["a_2"])

# now put it all together within the draw probability
data['lnprob_mass1_source_mass2_source_redshift_spin1spherical_spin2spherical'] = lnprob + lnjac_mass1_source + lnjac_mass2_source + lnjac_redshift + ln_jac_spin1 + ln_jac_spin2
del data['lnprob_mass1_source_mass2_source_redshift_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z']

#---

### add remaining draw probability
if args.verbose:
    print('    computing the remaining draw probabilities')
data["geocenter_time"] = np.zeros_like(data["mass1_source"])
data['lnprob_geocenter_time'] = np.ones_like(data['geocenter_time'])

#---

### write updated samples to disk

# format data into an array
header = sorted(data.keys())
data = np.transpose([data[key] for key in header])

# write to disk
if args.verbose:
    print('writing %d injections to : %s'%(len(data), args.target))
np.savetxt(args.target, data, header=','.join(header), comments='', delimiter=',')

for key, val in attrs.items():
    path = args.target + '-' + key
    if args.verbose:
        print('    writing %s=%f to : %s'%(key, val, path))
    with open(path, 'w') as obj:
        obj.write('%d'%val)
