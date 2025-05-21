#!/bin/bash

source mmms_shared_config.sh

POP_LABEL="multiPDB_betaSplit3_brokenG"
POP_FOLDER="conversion_scripts/${POP_LABEL}"
EOS_SAMPLES="LEC-2020"
COMPONENT="1"
SEED="--seed 7236"

EXTRA_EVENT_ARGS="--mass-column mass${COMPONENT}_source"

POP_ARGS="$POP_MAX_ARG --mtov-column notch_lowmass_scale"

LABEL="${EVENT_SAMPLES}+${POP_LABEL}+component${COMPONENT}"

# Assertions
[[ -f samples/${EVENT_SAMPLES}.csv.gz ]] || { echo "Missing event samples file"; exit 1; }
[[ -f ${POP_FOLDER}/population.csv.gz ]] || { echo "Missing pop samples file"; exit 1; }

mmms samples/${EVENT_SAMPLES}.csv.gz \
     ${POP_LABEL}.ini \
     ${POP_FOLDER}/population.csv.gz \
     ${EVENT_ARGS} \
     ${EXTRA_EVENT_ARGS} \
     ${POP_ARGS} \
     ${SEED} \
     --Verbose \
     1> testing_mmms/${LABEL}.out \
     2> testing_mmms/${LABEL}.err

# Plot command
mmms-plot \
    samples/${EVENT_SAMPLES}.csv.gz \
    ${EVENT_ARGS} \
    ${EXTRA_EVENT_ARGS} \
    ${POP_LABEL}.ini \
    ${POP_FOLDER}/population.csv.gz \
    ${POP_ARGS} \
    ${SEED} \
    --Verbose \
    --output-dir testing_mmms \
    --tag ${LABEL} \
    --figtype png --dpi 200


COMPONENT="2"
EXTRA_EVENT_ARGS="--mass-column mass${COMPONENT}_source"
LABEL="${EVENT_SAMPLES}+${POP_LABEL}+component${COMPONENT}"

# Assertions
[[ -f samples/${EVENT_SAMPLES}.csv.gz ]] || { echo "Missing event samples file"; exit 1; }
[[ -f ${POP_FOLDER}/population.csv.gz ]] || { echo "Missing pop samples file"; exit 1; }

mmms samples/${EVENT_SAMPLES}.csv.gz \
     ${POP_LABEL}.ini \
     ${POP_FOLDER}/population.csv.gz \
     ${EVENT_ARGS} \
     ${EXTRA_EVENT_ARGS} \
     ${POP_ARGS} \
     ${SEED} \
     --Verbose \
     1> testing_mmms/${LABEL}.out \
     2> testing_mmms/${LABEL}.err

# Plot command
mmms-plot \
    samples/${EVENT_SAMPLES}.csv.gz \
    ${EVENT_ARGS} \
    ${EXTRA_EVENT_ARGS} \
    ${POP_LABEL}.ini \
    ${POP_FOLDER}/population.csv.gz \
    ${POP_ARGS} \
    ${SEED} \
    --Verbose \
    --output-dir testing_mmms \
    --tag ${LABEL} \
    --figtype png --dpi 200

