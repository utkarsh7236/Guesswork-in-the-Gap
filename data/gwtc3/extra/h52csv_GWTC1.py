#!/usr/bin/env python3

"""a script to extract samples and standardize parameter names
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
    'm1_detector_frame_Msun' : 'mass1_detector',
    'm2_detector_frame_Msun' : 'mass2_detector',
    'luminosity_distance_Mpc' : 'luminosity_distance',
    'declination' : 'declination',
    'right_ascension' : 'right_ascension',
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
                print('    %s/%s -> %s'%(args.approximant, old, new))
            data[new] = obj[args.approximant][old][:]

    # luminosity_distance = data.pop('luminosity_distance') * MPC_CGS ### cosmo works in CGS, not Mpc
    luminosity_distance = data['luminosity_distance'] * MPC_CGS #TODO: Document change
    mass1_detector = data.pop('mass1_detector')
    mass2_detector = data.pop('mass2_detector')

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

    data['lnprob_declination'] = np.log(0.5*np.cos(data['declination']))
    data['lnprob_right_ascension'] = -np.log(2*np.pi)*np.ones_like(data['right_ascension'])

    #---
    data['geocenter_time'] = np.zeros_like(data['mass1_source']) # TODO: Document change
    data['lnprob_geocenter_time'] = np.zeros_like(data['geocenter_time']) # TODO: Document change

    ### write updated samples to disk

    # format data into an array
    header = sorted(data.keys())
    data = np.transpose([data[key] for key in header])

    if len(data) > args.max_num_samples:
        if args.verbose:
            print('    downselecting from %d to %d samples'%(len(data), args.max_num_samples))
        data = data[np.random.choice(np.arange(len(data)), replace=False, size=args.max_num_samples)]

    # write to disk
    event_list_name = os.path.basename(path).split('_')[0]
    nath = os.path.join(args.output_dir, os.path.basename(path).split('_')[0] + '.csv.gz')
    if args.verbose:
        print('    writing %d samples to : %s'%(len(data), nath))
    np.savetxt(nath, data, header=','.join(header), comments='', delimiter=',')

    with open(os.path.join(args.output_dir, 'event-list.txt'), 'r') as f:
        existing_entries = f.readlines()

    # If 'nath' is not found in the existing entries, open the file in append mode and add it
    if not any(event_list_name in line for line in existing_entries):
        with open(os.path.join(args.output_dir, 'event-list.txt'), 'a') as f:
            f.write(f'{event_list_name}'+ '.csv.gz\n')
