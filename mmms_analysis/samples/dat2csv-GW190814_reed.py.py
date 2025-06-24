#!/usr/bin/env python

import numpy as np
from numpy.lib import recfunctions as rfn

cols = ['mass_1_source', 'mass_2_source', 'luminosity_distance']
spin_cols = ['spin_1x', 'spin_1y', 'spin_1z', 'spin_2x', 'spin_2y', 'spin_2z']
header = ','.join(cols)
spin_header = ','.join(spin_cols)

### posterior

ans = np.genfromtxt('GW190814_posterior_samples.dat', names=True)
ans = ans[cols + spin_cols]
ans = rfn.append_fields(ans, ['mass1_source', 'mass2_source'],
                        [ans['mass_1_source'], ans['mass_2_source']],
                        usemask=False)

np.savetxt('GW190814_posterior_samples.csv.gz', ans, delimiter=',', comments='',
           header=header + ',' + spin_header + ',mass1_source,mass2_source')

np.savetxt('GW190814_posterior_samples.csv', ans, delimiter=',', comments='',
           header=header + ',' + spin_header + ',mass1_source,mass2_source')

### prior
wer = np.genfromtxt('GW190814_prior_samples.dat', names=True)
mc, q, dist = wer['mc'], wer['q'], wer['dist']
m1 = mc * (1 + q)**0.2 / q**0.6
m2 = m1 * q

np.savetxt('GW190814_prior_samples.csv', np.array(list(zip(m1, m2, dist))),
           delimiter=',', comments='', header=header)

np.savetxt('GW190814_prior_samples.csv.gz', np.array(list(zip(m1, m2, dist))),
           delimiter=',', comments='', header=header)
