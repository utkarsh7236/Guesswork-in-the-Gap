#!/bin/bash

POP="Updated-LVK-O3-sans-230529-PDB"

if [[ -f ${POP}.json ]]; then
    echo "[STATUS] ${POP}.json already exists, skipping scp step."
else
    echo "[STATUS] Fetching ${POP}.json from remote server..."
    scp utkarsh.mali@ldas-grid.ligo.caltech.edu:/home/michael.zevin/projects/o4/gw230529/analysis_results/populations/PowerlawDipBreak/production/no_230529_mass_h_result.json ./${POP}.json \
    || exit 1
fi

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

echo "[COMPLETED]"
