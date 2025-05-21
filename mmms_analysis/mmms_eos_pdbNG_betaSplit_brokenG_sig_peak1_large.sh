#!/bin/bash

source mmms_eos_shared_config.sh

POP_LABEL="pdbNG_betaSplit_brokenG_sig_peak1_large"
POP_FOLDER="conversion_scripts/${POP_LABEL}"
EOS_SAMPLES="LEC-2020"
COMPONENT="1"
SEED="--seed 7236"

# Assertions
[[ -f samples/${EVENT_SAMPLES}.csv.gz ]] || { echo "Missing event samples file"; exit 1; }
[[ -f samples/eos_population_mixtures/${POP_LABEL}_${EOS_SAMPLES}.csv.gz ]] || { echo "Missing combined eos pop samples file"; exit 1; }

LABEL="${EVENT_SAMPLES}+${POP_LABEL}_${EOS_SAMPLES}-${EOS_WEIGHT}+component${COMPONENT}"
EXTRA_EVENT_ARGS="--mass-column mass${COMPONENT}_source --spin-column spin${COMPONENT}_magnitude"
EXTRA_POP_ARGS="--pop-weight-column ${EOS_WEIGHT}"

mmms \
    samples/${EVENT_SAMPLES}.csv.gz \
    ${POP_LABEL}.ini \
    samples/eos_population_mixtures/${POP_LABEL}_${EOS_SAMPLES}.csv.gz \
    ${EVENT_ARGS} \
    ${EXTRA_EVENT_ARGS} \
    ${POP_ARGS} \
    ${EXTRA_POP_ARGS} \
    ${SEED} \
    --Verbose \
    1> testing_mmms/${LABEL}.out \
    2> testing_mmms/${LABEL}.err \
    || exit 1

mmms-plot \
    samples/${EVENT_SAMPLES}.csv.gz \
    $EVENT_ARGS \
    $EXTRA_EVENT_ARGS \
    ${POP_LABEL}.ini \
    samples/eos_population_mixtures/${POP_LABEL}_${EOS_SAMPLES}.csv.gz \
    $POP_ARGS \
    $EXTRA_POP_ARGS \
    $SEED \
    --Verbose \
    --output-dir testing_mmms \
    --tag ${LABEL} \
    --figtype png --dpi 200 \
|| exit 1

COMPONENT="2"
LABEL="${EVENT_SAMPLES}+${POP_LABEL}_${EOS_SAMPLES}-${EOS_WEIGHT}+component${COMPONENT}"
EXTRA_EVENT_ARGS="--mass-column mass${COMPONENT}_source --spin-column spin${COMPONENT}_magnitude"
EXTRA_POP_ARGS="--pop-weight-column ${EOS_WEIGHT}"

mmms \
    samples/${EVENT_SAMPLES}.csv.gz \
    ${POP_LABEL}.ini \
    samples/eos_population_mixtures/${POP_LABEL}_${EOS_SAMPLES}.csv.gz \
    ${EVENT_ARGS} \
    ${EXTRA_EVENT_ARGS} \
    ${POP_ARGS} \
    ${EXTRA_POP_ARGS} \
    ${SEED} \
    --Verbose \
    1> testing_mmms/${LABEL}.out \
    2> testing_mmms/${LABEL}.err \
    || exit 1

mmms-plot \
    samples/${EVENT_SAMPLES}.csv.gz \
    $EVENT_ARGS \
    $EXTRA_EVENT_ARGS \
    ${POP_LABEL}.ini \
    samples/eos_population_mixtures/${POP_LABEL}_${EOS_SAMPLES}.csv.gz \
    $POP_ARGS \
    $EXTRA_POP_ARGS \
    $SEED \
    --Verbose \
    --output-dir testing_mmms \
    --tag ${LABEL} \
    --figtype png --dpi 200 \
|| exit 1

