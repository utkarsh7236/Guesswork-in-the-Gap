#!/bin/bash

# Remove .out and .err files if they exist
# rm -f *.out *.err 

# Common population arguments
POP_MAX_ARG="--pop-max-num-samples $POP_MAX_NUM_SAMPLES"

EVENT_ARGS=""
EVENT_ARGS="$EVENT_ARGS --event-max-num-samples 10000"
EVENT_ARGS="$EVENT_ARGS --prior-column logprior"
EVENT_ARGS="$EVENT_ARGS --prior-is-log"
EVENT_ARGS="$EVENT_ARGS --m1-column mass1_source"
EVENT_ARGS="$EVENT_ARGS --m2-column mass2_source"
EVENT_ARGS="$EVENT_ARGS --d-column luminosity_distance"
EVENT_ARGS="$EVENT_ARGS --q-range 0.0 1.0"
EVENT_ARGS="$EVENT_ARGS --m-range 0.0 100.0"
EVENT_ARGS="$EVENT_ARGS --mc-range 0.0 20.0"
EVENT_ARGS="$EVENT_ARGS --d-range 0.0 10000.0"

POP_ARGS="$POP_MAX_ARG"
POP_ARGS="$POP_ARGS --mtov-column Mmax"
POP_ARGS="$POP_ARGS --rtov-column Rmax"
POP_ARGS="$POP_ARGS --pop-weight-is-log"

EOS_WEIGHT="logweight_PSR_GW"

POP_MAX_NUM_SAMPLES=1000
EOS_MAX_NUM_SAMPLES=5000
POP_MAX_ARG="--pop-max-num-samples $POP_MAX_NUM_SAMPLES"

EOS_NAME="LEC-2020"
EOS_SAMPLES="../samples/${EOS_NAME}"

POP="Updated-LVK-O3-sans-230529-PDB"
OUT="eos_population_mixtures/${EOS_NAME}_${POP}"


EVENT_SAMPLES="GW230529_Combined_PHM_highSpin"
COMPONENT="1"

# Ensure the input files exist
[[ -f ${POP}.csv.gz ]] || { echo "Missing population file: ${POP}.csv"; exit 1; }
[[ -f ${EOS_SAMPLES}.csv.gz ]] || { echo "Missing EOS file: ${EOS_SAMPLES}.csv"; exit 1; }

mmms-combine \
    --samples ${EOS_SAMPLES}.csv.gz ${EOS_MAX_NUM_SAMPLES} \
    --samples ${POP}.csv.gz ${POP_MAX_NUM_SAMPLES} \
    --outpath ${OUT}.csv.gz \
    ${SEED} \
    --verbose \

  # Define population labels
  POP_LABEL="LVK-O3-230529-PDB"
  SEED="--seed 7236"

  # EXTRA_EVENT_ARGS="--mass-column mass${COMPONENT}_source"
  # POP_ARGS="$POP_MAX_ARG --mtov-column notch_lowmass_scale"
  EXTRA_EVENT_ARGS="--mass-column mass${COMPONENT}_source --spin-column spin${COMPONENT}_magnitude"
  EXTRA_POP_ARGS="--pop-weight-column ${EOS_WEIGHT}"

  LABEL="${EVENT_SAMPLES}+${POP_LABEL}+${POP_PARAM}+${POP_VALUE}+component${COMPONENT}"

  # Assertions
  [[ -f ../samples/${EVENT_SAMPLES}.csv.gz ]] || { echo "Missing event samples file"; exit 1; }
  [[ -f eos_population_mixtures/${EOS_NAME}_${POP}.csv.gz ]] || { echo "Missing pop samples file"; exit 1; }
  [[ -f ../${POP_LABEL}.ini ]] || { echo "Missing gw-distribution initialization file"; exit 1; }

  FOLDER_NAME="${PWD##*/}"

  cd .. || { echo "Failed to cd .."; exit 1; }


echo "[STATUS] Running mmms for event ${EVENT_SAMPLES}, value ${POP_VALUE}, component ${COMPONENT}..."
LABEL="${EVENT_SAMPLES}+${POP_LABEL}+${POP_PARAM}+${POP_VALUE}+component${COMPONENT}"

# Assertions
[[ -f samples/${EVENT_SAMPLES}.csv.gz ]] || { echo "Missing event samples file"; exit 1; }
[[ -f ${FOLDER_NAME}/eos_population_mixtures/${EOS_NAME}_${POP}.csv.gz ]] || { echo "Missing pop samples file"; exit 1; }    
[[ -f ${POP_LABEL}.ini ]] || { echo "Missing gw-distribution initialization file"; exit 1; }

mmms \
samples/${EVENT_SAMPLES}.csv.gz \
${POP_LABEL}.ini \
${FOLDER_NAME}/eos_population_mixtures/${EOS_NAME}_${POP}.csv.gz \
${EVENT_ARGS} \
${EXTRA_EVENT_ARGS} \
${POP_ARGS} \
${EXTRA_POP_ARGS} \
${SEED} \
1> ${FOLDER_NAME}/${LABEL}.out \
2> ${FOLDER_NAME}/${LABEL}.err &