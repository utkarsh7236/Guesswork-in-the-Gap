#!/bin/bash

source mmms_shared_config.sh

EVENT_SAMPLES="gw230529_highSpin"
POP_LABEL="pdbNG_betaSplit_brokenG_tight_prior"
POP_FOLDER="conversion_scripts/${POP_LABEL}"
EOS_SAMPLES="LEC-2020"
COMPONENT="1"
SEED="--seed 7236"

EVENT_ARGS=""
EVENT_ARGS="$EVENT_ARGS --event-max-num-samples 10000"
EVENT_ARGS="$EVENT_ARGS --prior-column logprior"
EVENT_ARGS="$EVENT_ARGS --prior-is-log"
EVENT_ARGS="$EVENT_ARGS --m1-column mass1_source"
EVENT_ARGS="$EVENT_ARGS --m2-column mass2_source"
EVENT_ARGS="$EVENT_ARGS --d-column luminosity_distance"
EVENT_ARGS="$EVENT_ARGS --q-range 0.0 1.0"
EVENT_ARGS="$EVENT_ARGS --m-range 0.0 10.0"
EVENT_ARGS="$EVENT_ARGS --mc-range 0.0 10.0"
EVENT_ARGS="$EVENT_ARGS --d-range 0.0 10000.0"

EXTRA_EVENT_ARGS="--mass-column mass${COMPONENT}_source"

POP_ARGS="$POP_MAX_ARG --mtov-column notch_lowmass_scale"

LABEL="${EVENT_SAMPLES}+${POP_LABEL}+component${COMPONENT}"

# Assertions
[[ -f etc/${EVENT_SAMPLES}.csv.gz ]] || { echo "Missing event samples file"; exit 1; }
[[ -f ${POP_FOLDER}/population.csv.gz ]] || { echo "Missing pop samples file"; exit 1; }

mmms etc/${EVENT_SAMPLES}.csv.gz \
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
    etc/${EVENT_SAMPLES}.csv.gz \
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
[[ -f etc/${EVENT_SAMPLES}.csv.gz ]] || { echo "Missing event samples file"; exit 1; }
[[ -f ${POP_FOLDER}/population.csv.gz ]] || { echo "Missing pop samples file"; exit 1; }

mmms etc/${EVENT_SAMPLES}.csv.gz \
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
    etc/${EVENT_SAMPLES}.csv.gz \
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


