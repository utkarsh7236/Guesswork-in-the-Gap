#!/usr/bin/env python

import numpy as np
from numpy.lib import recfunctions as rfn
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u

# Define columns
cols = ['mass_1_source', 'mass_2_source', 'luminosity_distance']
spin_cols = ['spin_1x', 'spin_1y', 'spin_1z', 'spin_2x', 'spin_2y', 'spin_2z']
header_cols = cols + ['redshift']
header = ','.join(header_cols)
spin_header = ','.join(spin_cols)

### posterior

# Read posterior samples
ans = np.genfromtxt('GW190814_posterior_samples.dat', names=True)
ans = ans[cols + spin_cols]

# Compute redshift from luminosity distance
luminosity_distances = ans['luminosity_distance'] * u.Mpc
redshifts = np.array([z_at_value(cosmo.luminosity_distance, d) for d in luminosity_distances])

# Add logprior and renamed masses
ans = rfn.append_fields(ans,
                        ['mass1_source', 'mass2_source', 'logprior', 'redshift'],
                        [ans['mass_1_source'],
                         ans['mass_2_source'],
                         np.log(1 + 1e-7 * ans['luminosity_distance']),
                         redshifts],
                        usemask=False)

# Drop original mass_1_source and mass_2_source
ans = rfn.drop_fields(ans, ['mass_1_source', 'mass_2_source'])

# Save posterior CSV files
save_header = header + ',' + spin_header + ',mass1_source,mass2_source,logprior,redshift'
np.savetxt('GW190814_posterior_samplesTEST.csv', ans, delimiter=',', comments='', header=save_header)
np.savetxt('GW190814_posterior_samplesTEST.csv.gz', ans, delimiter=',', comments='', header=save_header)

# ### prior
#
# # Read prior samples
# wer = np.genfromtxt('GW190814_prior_samples.dat', names=True)
# mc, q, dist = wer['mc'], wer['q'], wer['dist']
# m1 = mc * (1 + q)**0.2 / q**0.6
# m2 = m1 * q
#
# # Compute redshift for prior samples
# luminosity_distances_prior = dist * u.Mpc
# redshifts_prior = np.array([z_at_value(cosmo.luminosity_distance, d) for d in luminosity_distances_prior])
#
# # Combine columns for saving
# prior_array = np.array(list(zip(m1, m2, dist, redshifts_prior)))
#
# # Save prior CSV files
# prior_header = 'mass_1_source,mass_2_source,luminosity_distance,redshift'
# np.savetxt('GW190814_prior_samples.csv', prior_array, delimiter=',', comments='', header=prior_header)
# np.savetxt('GW190814_prior_samples.csv.gz', prior_array, delimiter=',', comments='', header=prior_header)
