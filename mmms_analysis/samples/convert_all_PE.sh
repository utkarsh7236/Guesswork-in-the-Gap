#!/bin/bash

FOLDER="../../data/events_of_interest/"

GW230529="IGWN-GWTC4-manual-download-GW230529_181500.h5"
GW230529_waveform="Combined_PHM_highSpin"

#./gw230529-hdf2csv \
#    "${FOLDER}${GW230529}" \
#    "GW230529_${GW230529_waveform}.csv.gz" \
#    --root ${GW230529_waveform} || {
#  echo "Error converting $GW230529"
#  exit 1
#}
#
#GW230529="IGWN-GWTC4-manual-download-GW230529_181500.h5"
#GW230529_waveform="Combined_PHM_lowSecondarySpin"
#
#./gw230529-hdf2csv \
#    "${FOLDER}${GW230529}" \
#    "GW230529_${GW230529_waveform}.csv.gz" \
#    --root ${GW230529_waveform} || {
#  echo "Error converting $GW230529"
#  exit 1
#}
#GW190425="IGWN-GWTC2p1-v2-GW190425_081805_PEDataRelease_mixed_nocosmo.h5"
#GW190425_waveform="C01:IMRPhenomPv2_NRTidal:HighSpin"
#
#./alt-hdf2csv \
#    "${FOLDER}${GW190425}" \
#    "GW190425_${GW190425_waveform}.csv.gz" \
#    --root ${GW190425_waveform} || {
#  echo "Error converting $GW190425"
#  exit 1
#}
#GW190814="IGWN-GWTC2p1-v2-GW190814_211039_PEDataRelease_mixed_nocosmo.h5"
#GW190814_waveform="C01:IMRPhenomXPHM"
#./alt-hdf2csv \
#    "${FOLDER}${GW190814}" \
#    "GW190814_${GW190814_waveform}.csv.gz" \
#    --root ${GW190814_waveform} || {
#  echo "Error converting $GW190814"
#  exit 1
#}
#GW200115="IGWN-GWTC3p0-v2-GW200115_042309_PEDataRelease_mixed_nocosmo.h5"
#GW200115_waveform="C01:IMRPhenomNSBH:HighSpin"
#./alt-hdf2csv \
#    "${FOLDER}${GW200115}" \
#    "GW200115_${GW200115_waveform}.csv.gz" \
#    --root ${GW200115_waveform} || {
#  echo "Error converting $GW200115"
#  exit 1
#}
#GW200105="IGWN-GWTC3p0-v2-GW200105_162426_PEDataRelease_mixed_nocosmo.h5"
#GW200105_waveform="C01:IMRPhenomXPHM"
#./alt-hdf2csv \
#    "${FOLDER}${GW200105}" \
#    "GW200105_${GW200105_waveform}.csv.gz" \
#    --root ${GW200105_waveform} || {
#  echo "Error converting $GW200105"
#  exit 1
#}
#GW190917="IGWN-GWTC2p1-v2-GW190917_114630_PEDataRelease_mixed_nocosmo.h5"
#GW190917_waveform="C01:IMRPhenomXPHM"
#./alt-hdf2csv \
#    "${FOLDER}${GW190917}" \
#    "GW190917_${GW190917_waveform}.csv.gz" \
#    --root ${GW190917_waveform} || {
#  echo "Error converting $GW190917"
#  exit 1
#}
## TODO: GW170817
#GW170817="GW170817_GWTC-1.hdf5"
#GW170817_waveform="IMRPhenomPv2NRT_highSpin_posterior"
#./gw170817-hdf2csv \
#    "${FOLDER}${GW170817}" \
#    "GW170817_${GW170817_waveform}.csv.gz" \
#    --root ${GW170817_waveform} || {
#  echo "Error converting $GW170817"
#  exit 1
#}
