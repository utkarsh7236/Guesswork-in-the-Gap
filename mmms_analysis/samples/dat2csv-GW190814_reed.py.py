#!/usr/bin/env python

import numpy as np

cols = ['mass_1_source', 'mass_2_source', 'luminosity_distance']
spin_cols = ['spin_1x', 'spin_1y', 'spin_1z', 'spin_2x', 'spin_2y', 'spin_2z']
header = ','.join(cols)
spin_header = ','.join(spin_cols)

### posterior

ans = np.genfromtxt('GW190814_posterior_samples.dat', names=True)
ans = ans[cols+spin_cols]

np.savetxt('GW190814_posterior_samples.csv.gz', ans, delimiter=',', comments='', header=header+','+spin_header)

### prior

wer = np.genfromtxt('GW190814_prior_samples.dat', names=True)
mc = wer['mc']
q = wer['q']
dist = wer['dist']

m1 = mc * (1+q)**0.2 / q**0.6
m2 = m1 * q

np.savetxt('GW190814_prior_samples.csv', np.array(zip(m1, m2, dist)), delimiter=',', comments='', header=header)
