#!/bin/bash

EVENT_SAMPLES="Prod-BBH-HighSpin1-HighSpin2"
POP_LABEL="pdbNG_betaSplit_brokenG"
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

POP_ARGS=""
POP_ARGS="$POP_ARGS --pop-max-num-samples 100"
POP_ARGS="$POP_ARGS --mtov-column notch_lowmass_scale"

LABEL="${EVENT_SAMPLES}+${POP_LABEL}+component${COMPONENT}"

# Assertions
[[ -f ${EVENT_SAMPLES}.csv.gz ]] || { echo "Missing event samples file"; exit 1; }
[[ -f population.csv.gz ]] || { echo "Missing pop samples file"; exit 1; }

mmms ${EVENT_SAMPLES}.csv.gz \
     ${POP_LABEL}.ini \
     population.csv.gz \
     ${EVENT_ARGS} \
     ${EXTRA_EVENT_ARGS} \
     ${POP_ARGS} \
     ${SEED} \
     --Verbose \


