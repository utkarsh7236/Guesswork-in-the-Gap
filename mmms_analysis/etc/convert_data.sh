#!/bin/bash

# gather the required input files for the analysis
# Reed Essick (reed.essick@gmail.com)

#-------------------------------------------------

# grab EoS posteriors

EOSS=""

# the file used in O3 NSBH detection paper
EOS="LEC-mtov-rtov"
echo $EOS

# register this EoS
EOSS="$EOSS $EOS"

# construct a more complete dataset so we can select which astro logweights to include
EOS="LEC-2020"
echo $EOS

# preform matching experiment with LEC-mtov-rtov.csv.gz from the O3 NSBH detection paper to extract Mtov and Rtov
./lec-2020-combine \
|| exit 1

# this creates the following file
#   LEC-2020.csv.gz

# register this EoS
EOSS="$EOSS $EOS"

#-------------------------------------------------

# grab the population posteriors

POPS=""
# register this population to be combined with EoS posterior(s)
POPS="$POPS Default-PE"

#------------------------

### LVK O3b PDB model

POP="Prelim-LVK-O3-230529-PDB"
echo $POP

# convert from json to csv
./lvk-o3b-pdb-json2csv \
    ${POP}.json \
    ${POP}.csv.gz \
    --verbose \
|| exit 1

# register this population
POPS="$POPS $POP"

#-----------

POP="LVK-O3-230529-PDB"
echo $POP

# convert from json to csv
./lvk-o3b-pdb-json2csv \
    ${POP}.json \
    ${POP}.csv.gz \
    --verbose \
|| exit 1

# register this population to be combined with EoS posterior(s)
POPS="$POPS $POP"

#-----------

# convert from json to csv but force this to act as if there is a mass-independent spin distribution
./lvk-o3b-pdb-json2csv \
    ${POP}.json \
    ${POP}-forced.csv.gz \
    --force-single-spin-magnitude-distribution \
    --verbose \
|| exit 1

# register population
POPS="$POPS ${POP}-forced"

#------------------------

POP="Updated-LVK-O3-with-230529-PDB"
echo $POP

# convert from json to csv
./lvk-o3b-pdb-json2csv \
    ${POP}.json \
    ${POP}.csv.gz \
    --verbose \
|| exit 1

# register this population to be combined with EoS posterior(s)
POPS="$POPS $POP"

# convert from json to csv but force this to act as if there is a mass-independent spin distribution
./lvk-o3b-pdb-json2csv \
    ${POP}.json \
    ${POP}-forced.csv.gz \
    --force-single-spin-magnitude-distribution \
    --verbose \
|| exit 1

# register population
POPS="$POPS ${POP}-forced"

#-----------

POP="Updated-LVK-O3-sans-230529-PDB"
echo $POP

# convert from json to csv
./lvk-o3b-pdb-json2csv \
    ${POP}.json \
    ${POP}.csv.gz \
    --verbose \
|| exit 1

# register this population to be combined with EoS posterior(s)
POPS="$POPS $POP"

# convert from json to csv but force this to act as if there is a mass-independent spin distribution
./lvk-o3b-pdb-json2csv \
    ${POP}.json \
    ${POP}-forced.csv.gz \
    --force-single-spin-magnitude-distribution \
    --verbose \
|| exit 1

# register population
POPS="$POPS ${POP}-forced"

#------------------------

### Farah+2022 PDB models

for MODEL in "g" "h" "i"
do

    POP="Farah2022-${MODEL}-PDB"
    echo $POP

    # convert from json to csv
    ./farah2022-pdb-json2csv \
        ${POP}.json \
        ${POP}.csv.gz \
        --verbose \
    || exit 1

    # register this population to be combined with EoS posterior(s)
    POPS="$POPS $POP"

done

#------------------------

# now combine with EoS posterior(s)

for POP in $POPS
do

    # combine population and EoS samples into a single file
    for EOS in $EOSS
    do

        mmms-combine \
            --samples ${EOS}.csv 1000 \
            --samples ${POP}.csv 1000 \
            --outpath ${POP}+${EOS}.csv.gz \
            --seed 123 \
            --verbose \
        || exit 1

    done
done

#-------------------------------------------------

# grab single-event PE samples based on: https://ldas-jobs.ligo.caltech.edu/~pe.o4/O4a/#s230529ay
# then convert each to a CSV

echo "grabbing single-event posterior samples"

#------------------------

# BBH waveform : Exp6 (wide spin1 prior, narrow spin2 prior)
SAMP="BBH-HighSpin1-LowSpin2"
echo $SAMP

# convert from hdf to csv
./hdf2csv \
    ${SAMP}.hdf5 \
    ${SAMP}.csv.gz \
    --verbose \
|| exit 1

#------------------------

# BBH waveform: Prod2 (wide spin1 and spin2 priors)
SAMP="BBH-HighSpin1-HighSpin2"
echo $SAMP

# convert from hdf to csv
./hdf2csv \
    ${SAMP}.hdf5 \
    ${SAMP}.csv.gz \
    --verbose \
|| exit 1

#------------------------

# unknown waveform and priors; what Amanda Farah used in her preliminary PDB fit
SAMP="Unknown-Prelim"
echo $SAMP

# convert from hdf to csv

./alt-hdf2csv \
    --root '1685387158_G408702_data0_1369419318-7460938_analysis_L1_merge_result' \
    ${SAMP}.hdf5 \
    ${SAMP}.csv.gz \
    --verbose \
|| exit 1

#-------------------------------------------------
#
# PRODUCTION PE
#
#-------------------------------------------------

SAMP="Prod-BBH-HighSpin1-HighSpin2"
echo $SAMP

# convert from hdf to csv

./alt-hdf2csv \
    --root 'combined_imrphm_high_spin' \
    ${SAMP}.hdf5 \
    ${SAMP}.csv.gz \
    --verbose \
|| exit 1

#------------------------

SAMP="Prod-BBH-HighSpin1-LowSpin2"
echo $SAMP

# convert from hdf to csv

./alt-hdf2csv \
    --root 'combined_imrphm_low_spin' \
    ${SAMP}.hdf5 \
    ${SAMP}.csv.gz \
    --verbose \
|| exit 1

#------------------------

SAMP="Prod-BBH-LowSpin1-LowSpin2"
echo $SAMP

# convert from hdf to csv

./alt-hdf2csv \
    --root 'Prod14' \
    ${SAMP}.hdf5 \
    ${SAMP}.csv.gz \
    --verbose \
|| exit 1
