#!/bin/bash

POP="Updated-LVK-O3-sans-230529-PDB"
<<<<<<< HEAD
=======

>>>>>>> 69632e7049a6a7dae9fe949846945086c5f56c45
if [[ -f ${POP}.json ]]; then
    echo "[STATUS] ${POP}.json already exists, skipping scp step."
else
    echo "[STATUS] Fetching ${POP}.json from remote server..."
    scp utkarsh.mali@ldas-grid.ligo.caltech.edu:/home/michael.zevin/projects/o4/gw230529/analysis_results/populations/PowerlawDipBreak/production/no_230529_mass_h_result.json ./${POP}.json \
    || exit 1
fi

<<<<<<< HEAD

# convert from json to csv but force this to act as if there is a mass-independent spin distribution
./lvk-o3b-pdb-json2csv \
    ${POP}.json \
    ${POP}-forced.csv.gz \
    --force-single-spin-magnitude-distribution \
    --verbose \
    || exit 1

# convert from json to csv but force this to act as if there is a mass-independent spin distribution
./lvk-o3b-pdb-json2csv \
    ${POP}.json \
    ${POP}-forced.csv.gz \
    --force-single-spin-magnitude-distribution \
    --verbose \
    || exit 1

POP="Farah2022-g-PDB"

if [[ -f ${POP}.json ]]; then
    echo "[STATUS] ${POP}.json already exists, skipping scp step."
else
    echo "[STATUS] Fetching ${POP}.json from remote server..."
    scp utkarsh.mali@ldas-grid.ligo.caltech.edu:/home/amanda.farah/projects/O3/population_runs/chips_dip/binned_pairing/mbreak_is_gammalow/result/o1o2o3a_mass_g_iid_mag_iid_tilt_powerlaw_redshift_result.json ${POP}.json \
    || exit 1
fi

# convert from json to csv
./farah2022-pdb-json2csv \
    ${POP}.json \
    ${POP}.csv.gz \
    --verbose \
|| exit 1

=======
# convert from json to csv
./lvk-o3b-pdb-json2csv \
    ${POP}.json \
    ${POP}.csv.gz \
    --verbose \
    || exit 1

echo "[STATUS] Converted ${POP}.json to ${POP}.csv.gz"

# convert from json to csv but force this to act as if there is a mass-independent spin distribution
./lvk-o3b-pdb-json2csv \
    ${POP}.json \
    ${POP}-forced.csv.gz \
    --force-single-spin-magnitude-distribution \
    --verbose \
    || exit 1
>>>>>>> 69632e7049a6a7dae9fe949846945086c5f56c45

echo "[COMPLETED]"
