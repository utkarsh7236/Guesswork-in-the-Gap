#!/usr/bin/env python

"""a simple script to convert PE samples into a standard format.
Only reads detector-frame values from the PE samples and instantiates our own a flat LambdaCDM cosmology \
to convert to source-frame parameters

-- Authored by Reed Essick
-- Modified by Utkarsh Mali
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import os

import numpy as np
import h5py

from argparse import ArgumentParser

### non-standard libraries
from conversion_cosmology import Cosmology
### Planck 2018 Cosmology (Table1 in arXiv:1807.06209)
PLANCK_2018_Ho = 2.1816926176539463e-18 ### CGS
PLANCK_2018_OmegaMatter = 0.3158
PLANCK_2018_OmegaLambda = 1. - PLANCK_2018_OmegaMatter
PLANCK_2018_OmegaRadiation = 0.
# PLANCK_2018_OmegaKappa = 0.
PC_SI = 3.085677581491367e+16
MPC_SI = PC_SI * 1e6
MPC_CGS = MPC_SI * 1e2

#-------------------------------------------------

names = {
    'mass_1' : 'mass1_detector',
    'mass_2' : 'mass2_detector',
    'luminosity_distance' : 'luminosity_distance',
    'spin_1x' : 'spin1x',
    'spin_1y' : 'spin1y',
    'spin_1z' : 'spin1z',
    'spin_2x' : 'spin2x',
    'spin_2y' : 'spin2y',
    'spin_2z' : 'spin2z',
    'dec' : 'declination',
    'ra' : 'right_ascension',
    'geocent_time' : 'geocenter_time',
    'log_likelihood' : 'loglikelihood',
}

#-------------------------------------------------

parser = ArgumentParser(description=__doc__)

parser.add_argument('approximant', type=str)
parser.add_argument('paths', nargs='+', type=str)

parser.add_argument('--max-num-samples', default=10000, type=int,
    help='the maximum number of samples per event that will be retained. DEFAULT=10000')
parser.add_argument('--seed', default=None, type=int)

parser.add_argument('--Ho', default=PLANCK_2018_Ho, type=float,
    help='Hubble parameter at z=0 in CGS. DEFAULT=%.6e'%PLANCK_2018_Ho)
parser.add_argument('--OmegaLambda', default=PLANCK_2018_OmegaLambda, type=float,
    help='DEFAULT=%.6f'%PLANCK_2018_OmegaLambda)
parser.add_argument('--OmegaMatter', default=PLANCK_2018_OmegaMatter, type=float,
    help='DEFAULT=%.6f'%PLANCK_2018_OmegaMatter)
parser.add_argument('--OmegaRadiation', default=PLANCK_2018_OmegaRadiation, type=float,
    help='DEFAULT=%.6e'%PLANCK_2018_OmegaRadiation)

parser.add_argument('-v', '--verbose', default=False, action='store_true')

parser.add_argument('-o', '--output-dir', default='.', type=str)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

#-------------------------------------------------

if args.seed is not None:
    if args.verbose:
        print('setting numpy.random.seed=%d'%args.seed)
    np.random.seed(args.seed)

#-------------------------------------------------

if args.verbose:
    print('''instantiating cosmology:
    Ho=%.6e
    OmegaMatter=%.6e
    OmegaRadiation=%.6e
    OmegaLambda=%.6e'''%(args.Ho, args.OmegaMatter, args.OmegaRadiation, args.OmegaLambda))
cosmo = Cosmology(args.Ho, args.OmegaMatter, args.OmegaRadiation, args.OmegaLambda)

#-------------------------------------------------

for path in args.paths:
    if args.verbose:
        print('loading : '+path)

    ### load detector-frame samples
    data = dict()
    with h5py.File(path, 'r') as obj:
        for old, new in names.items():
            if args.verbose:
                print('    %s/posterior_samples/%s -> %s'%(args.approximant, old, new))
            data[new] = obj[args.approximant]['posterior_samples'][old][:]

#    luminosity_distance = data.pop('luminosity_distance') * MPC_CGS ### cosmo works in CGS, not Mpc
    luminosity_distance = data['luminosity_distance'] * MPC_CGS # don't remove this from the dictionary
    mass1_detector = data.pop('mass1_detector')
    mass2_detector = data.pop('mass2_detector')
    if data['luminosity_distance'].size > 10000:
        pass
    else:
        print(f'Please check the input file:\n{path}.')
        raise ValueError("The number of samples is less than 10000")

    #---

    if args.verbose:
        print('    converting detector-frame paramters to source-frame parameters')

    ### extend cosmology to required redshift

    cosmo.extend(max_DL=np.max(luminosity_distance))

    ### compute redshift and source-frame properties
    data['redshift'] = cosmo.DL2z(luminosity_distance)
    data['mass1_source'] = mass1_detector / (1.+data['redshift'])
    data['mass2_source'] = mass2_detector / (1.+data['redshift'])

    #---

    ### compute the prior
    if args.verbose:
        print('    computing the (default) parameter estimation prior')

    data['lnprob_mass1_source'] = 1 + data['redshift']
    data['lnprob_mass2_source'] = 1 + data['redshift']

    data['lnprob_redshift'] = 2*np.log(luminosity_distance) \
        + np.log(cosmo.z2Dc(data['redshift']) + (1+data['redshift'])*cosmo.dDcdz(data['redshift']))

    spin_sqrd = data['spin1x']**2 + data['spin1y']**2 + data['spin1z']**2
    data['lnprob_spin1x_spin1y_spin1z'] = -np.log(4*np.pi*spin_sqrd)

    spin_sqrd = data['spin2x']**2 + data['spin2y']**2 + data['spin2z']**2
    data['lnprob_spin2x_spin2y_spin2z'] = -np.log(4*np.pi*spin_sqrd)

    data['lnprob_declination'] = np.log(0.5*np.cos(data['declination']))
    data['lnprob_right_ascension'] = -np.log(2*np.pi)*np.ones_like(data['right_ascension'])

    data['lnprob_geocenter_time'] = np.zeros_like(data['geocenter_time']) ### constant, but unknown normalization

    #---

    ### write updated samples to disk

    # format data into an array
    header = sorted(data.keys())
    data = np.transpose([data[key] for key in header])

    if len(data) > args.max_num_samples:
        if args.verbose:
            print('    downselecting from %d to %d samples'%(len(data), args.max_num_samples))
        data = data[np.random.choice(np.arange(len(data)), replace=False, size=args.max_num_samples)]

    # write to disk
    event_list_name = '_'.join(os.path.basename(path)[:-2].split('-')[-1].split('_')[:2])
    nath = event_list_name
    nath = os.path.join(args.output_dir, nath + '.csv.gz')
    if args.verbose:
        print('    writing %d samples to : %s'%(len(data), nath))
    np.savetxt(nath, data, header=','.join(header), comments='', delimiter=',')

    with open(os.path.join(args.output_dir, 'event-list.txt'), 'r') as f:
        existing_entries = f.readlines()

    # If 'nath' is not found in the existing entries, open the file in append mode and add it
    if not any(f'{event_list_name}'+ '.csv.gz' in line for line in existing_entries):
        with open(os.path.join(args.output_dir, 'event-list.txt'), 'a') as f:
            f.write(f'{event_list_name}'+ '.csv.gz\n')