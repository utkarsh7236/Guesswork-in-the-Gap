#!/bin/bash

FOLDER="../../data/events_of_interest/"

GW230529="IGWN-GWTC4-manual-download-GW230529_181500.h5"
GW230529_waveform="Combined_PHM_highSpin"

./alt-hdf2csv \
    "${FOLDER}${GW230529}" \
    "GW230529_${GW230529_waveform}.csv.gz" \
    --root ${GW230529_waveform} || {
  echo "Error converting $GW230529"
  exit 1
}

GW230529="IGWN-GWTC4-manual-download-GW230529_181500.h5"
GW230529_waveform="Combined_PHM_lowSecondarySpin"

./alt-hdf2csv \
    "${FOLDER}${GW230529}" \
    "GW230529_${GW230529_waveform}.csv.gz" \
    --root ${GW230529_waveform} || {
  echo "Error converting $GW230529"
  exit 1
}

