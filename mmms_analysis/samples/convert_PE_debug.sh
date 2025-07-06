#!/bin/bash

FOLDER="../../data/events_of_interest/"

GW190814="GW190814_posterior_samples.h5"
GW190814_waveform="C01:SEOBNRv4_ROM_NRTidalv2_NSBH"

./alt-hdf2csv-debug-pe \
    "${FOLDER}${GW190814}" \
    "GW190814_${GW190814_waveform}testPE.csv.gz" \
    --root ${GW190814_waveform} || {
  echo "Error converting $GW190814"
  exit 1
}

#GW190814="IGWN-GWTC2p1-v2-GW190814_211039_PEDataRelease_mixed_nocosmo.h5"
#GW190814_waveform="C01:IMRPhenomXPHM"
#./alt-hdf2csv-debug \
#    "${FOLDER}${GW190814}" \
#    "GW190814_${GW190814_waveform}const_jac.csv.gz" \
#    --root ${GW190814_waveform} || {
#  echo "Error converting $GW190814"
#  exit 1
#}
#GW190814="IGWN-GWTC2p1-v2-GW190814_211039_PEDataRelease_mixed_nocosmo.h5"
#GW190814_waveform="C01:Mixed"
#./alt-hdf2csv-debug \
#    "${FOLDER}${GW190814}" \
#    "GW190814_${GW190814_waveform}const_jac.csv.gz" \
#    --root ${GW190814_waveform} || {
#  echo "Error converting $GW190814"
#  exit 1
#}