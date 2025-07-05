#!/bin/bash

# Remove all files that end with .out, .err, or .csv.gz in the current directory
echo "[STATUS] Cleaning up old output files..."
rm -f *.out *.err *.csv.gz
echo "[STATUS] Old output files cleaned."

# Import shared config
source ../mmms_shared_config.sh

# Fixing pop samples count for now
POP_MAX_NUM_SAMPLES=100
POP_MAX_ARG="--pop-max-num-samples $POP_MAX_NUM_SAMPLES"

# Define events to use along with waveform type
ALL_EVENTS=(
  "GW190814_C01:IMRPhenomXPHM|2"
)

POP_PARAM="gamma_low"
POP_VALUES=(2.2 2.4 2.6 2.8 3.0 3.2 3.6 4.0)   # Must be a float

# Running conversion script, you need to change the value of pop_param in the conversion script
for POP_VALUE in "${POP_VALUES[@]}"; do
  # THE MOST IMPORTANT CHANGING VARIABLE NEEDS TO BE THE FIRST ELEMENT IN LIST!
  echo "[STATUS] Running conversion for POP_VALUE=${POP_VALUE}..."
  python3 convert_multiPDB_betaSplit_brokenG.py \
    --pop_param "$POP_PARAM" eta_low \
    --pop_value "$POP_VALUE" 50 &
done

# Wait for all backgrounded conversions to finish
wait

printf "\n\n[STATUS] Completed conversion scripts, running mmms now...\n"

for ENTRY in "${ALL_EVENTS[@]}"; do
  # Split into key and value parts
  EVENT_SAMPLES=${ENTRY%%|*}
  VALUE=${ENTRY#*|}
  for COMPONENT in $VALUE; do

  # Run individual mmms

  # Define population labels
  POP_LABEL="multiPDB_betaSplit_brokenG"
  SEED="--seed 7236"

  EXTRA_EVENT_ARGS="--mass-column mass${COMPONENT}_source"
  POP_ARGS="$POP_MAX_ARG --mtov-column notch_lowmass_scale"

  LABEL="${EVENT_SAMPLES}+${POP_LABEL}+${POP_PARAM}+${POP_VALUE}+component${COMPONENT}"

  # Assertions
  [[ -f ../samples/${EVENT_SAMPLES}.csv.gz ]] || { echo "Missing event samples file"; exit 1; }
  [[ -f population${POP_VALUE}.csv.gz ]] || { echo "Missing pop samples file"; exit 1; }
  [[ -f ../${POP_LABEL}.ini ]] || { echo "Missing gw-distribution initialization file"; exit 1; }

  FOLDER_NAME="${PWD##*/}"

  cd .. || { echo "Failed to cd .."; exit 1; }

  # Run mmms for each POP_VALUE in parallel
  for POP_VALUE in "${POP_VALUES[@]}"; do
    LABEL="${EVENT_SAMPLES}+${POP_LABEL}+${POP_PARAM}+${POP_VALUE}+component${COMPONENT}"

    # Assertions
    [[ -f samples/${EVENT_SAMPLES}.csv.gz ]] || { echo "Missing event samples file"; exit 1; }
    [[ -f ${FOLDER_NAME}/population${POP_VALUE}.csv.gz ]] || { echo "Missing pop samples file for POP_VALUE=${POP_VALUE}"; exit 1; }
    [[ -f ${POP_LABEL}.ini ]] || { echo "Missing gw-distribution initialization file"; exit 1; }

    mmms samples/${EVENT_SAMPLES}.csv.gz \
         ${POP_LABEL}.ini \
         ${FOLDER_NAME}/population${POP_VALUE}.csv.gz \
         ${EVENT_ARGS} \
         ${EXTRA_EVENT_ARGS} \
         ${POP_ARGS} \
         ${SEED} \
         1> ${FOLDER_NAME}/${LABEL}.out \
         2> ${FOLDER_NAME}/${LABEL}.err &
  done

  cd "${FOLDER_NAME}" || { echo "Failed to cd back to original dir"; exit 1; }

  done
done

wait  # wait for all background jobs to finish

printf " \n[COMPLETED]\n "
