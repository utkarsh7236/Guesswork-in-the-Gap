#!/bin/bash

EOS_WEIGHT="logweight_PSR_GW_Xray"
EVENT_SAMPLES="Prod-BBH-HighSpin1-HighSpin2"
POP_SAMPLES="Farah2022-g-PDB"
EOS_SAMPLES="LEC-2020"
COMPONENT="1"
SEED="--seed 7236"

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

POP_ARGS=""

POP_ARGS="$POP_ARGS --pop-max-num-samples 10000" # 10000 is full
POP_ARGS="$POP_ARGS --mtov-column Mmax"
POP_ARGS="$POP_ARGS --rtov-column Rmax"
POP_ARGS="$POP_ARGS --pop-weight-is-log"

LABEL="${EVENT_SAMPLES}+${POP_SAMPLES}+${EOS_SAMPLES}-${EOS_WEIGHT}+component${COMPONENT}"
EXTRA_EVENT_ARGS="--mass-column mass${COMPONENT}_source --spin-column spin${COMPONENT}_magnitude"
EXTRA_POP_ARGS="--pop-weight-column ${EOS_WEIGHT}"

# Assertions
[[ -f etc/${EVENT_SAMPLES}.csv.gz ]] || { echo "Missing event samples file"; exit 1; }
[[ -f etc/${POP_SAMPLES}+${EOS_SAMPLES}.csv.gz ]] || { echo "Missing pop samples file"; exit 1; }

mmms \
    etc/${EVENT_SAMPLES}.csv.gz \
    ${POP_SAMPLES}.ini \
    etc/${POP_SAMPLES}+${EOS_SAMPLES}.csv.gz \
    $EVENT_ARGS \
    $EXTRA_EVENT_ARGS \
    $POP_ARGS \
    $EXTRA_POP_ARGS \
    $SEED \
    --Verbose \
    1> testing_mmms/${LABEL}.out \
    2> testing_mmms/${LABEL}.err \
    || exit 1

mmms-plot \
    etc/${EVENT_SAMPLES}.csv.gz \
    $EVENT_ARGS \
    $EXTRA_EVENT_ARGS \
    ${POP_SAMPLES}.ini \
    etc/${POP_SAMPLES}+${EOS_SAMPLES}.csv.gz \
    $POP_ARGS \
    $EXTRA_POP_ARGS \
    $SEED \
    --Verbose \
    --output-dir testing_mmms \
    --tag ${LABEL} \
    --figtype png --dpi 200 \
|| exit 1


