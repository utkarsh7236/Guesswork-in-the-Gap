#!/bin/bash

# Remove all files that end with .out, .err, or .csv.gz in the current directory
echo "[STATUS] Cleaning up old output files..."
rm -f *.out *.err
echo "[STATUS] Old output files cleaned."

# Import shared config
source ../mmms_shared_config.sh

# Fixing pop samples count for now
POP_MAX_NUM_SAMPLES=100
POP_MAX_ARG="--pop-max-num-samples $POP_MAX_NUM_SAMPLES"

# Define events to use along with waveform type
ALL_EVENTS=(
  # "GW230529_Combined_PHM_highSpin|1"
  # "GW190425_C01:IMRPhenomPv2_NRTidal:HighSpin|1 2"
  "GW190814_C01:IMRPhenomXPHM|2"
  # "GW190814_C01:MixedTEST|2"
  # "GW190814_posterior_samplesTEST|2"
  # "GW190917_C01:IMRPhenomXPHM|1 2"
  # "GW200105_C01:IMRPhenomXPHM|1 2"
  # "GW200115_C01:IMRPhenomNSBH:HighSpin|2"
)

POP_PARAM="gamma_low"

POP_VALUES=(2.1 2.12 2.14 2.16 2.18 2.2 2.22 2.24 2.26 2.28 2.3 2.32 2.34 2.36 2.38 2.4 2.42 2.44 2.46 2.48 2.5 2.52 2.54 2.56 2.58 2.6 2.62 2.64 2.66 2.68 2.7 2.72 2.74 2.76 2.78 2.8 2.82 2.84 2.86 2.88 2.9 2.92 2.94 2.96 2.98 3.0 3.02 3.04 3.06 3.08 3.1 3.12 3.14 3.16 3.18 3.2 3.22 3.24 3.26 3.28 3.3 3.32 3.34 3.36 3.38 3.4 3.42 3.44 3.46 3.48 3.5)

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

printf "\n\n[STATUS] Completed conversion scripts, running PDB mmms now...\n"

for ENTRY in "${ALL_EVENTS[@]}"; do
  # Split into key and value parts
  EVENT_SAMPLES=${ENTRY%%|*}
  VALUE=${ENTRY#*|}
  for COMPONENT in $VALUE; do

  # Run individual mmms

  # Define population labels
  POP_LABEL="Farah2022-i-PDB"
  # POP_LABEL="multiPDB_betaSplit_brokenG"
  SEED="--seed 7236"

  # EXTRA_EVENT_ARGS="--mass-column mass${COMPONENT}_source"
  EXTRA_EVENT_ARGS="--mass-column mass${COMPONENT}_source"
  POP_ARGS="$POP_MAX_ARG --mtov-column notch_lowmass_scale"

  LABEL="${EVENT_SAMPLES}+${POP_LABEL}+${POP_PARAM}+${POP_VALUE}+component${COMPONENT}"

  FOLDER_NAME="${PWD##*/}"

  cd .. || { echo "Failed to cd .."; exit 1; }

  # Run mmms for each POP_VALUE in parallel
  for POP_VALUE in "${POP_VALUES[@]}"; do
    LABEL="${EVENT_SAMPLES}+${POP_LABEL}+${POP_PARAM}+${POP_VALUE}+component${COMPONENT}"

    # echo "${FOLDER_NAME}/population${POP_VALUE}.csv.gz"
    echo "[STATUS] Running mmms for ${LABEL}..."

    # Assertions
    [[ -f samples/${EVENT_SAMPLES}.csv.gz ]] || { echo "Missing event samples file"; exit 1; }
    [[ -f ${FOLDER_NAME}/PDBpopulation${POP_VALUE}.csv.gz ]] || { echo "Missing pop samples file for POP_VALUE=${POP_VALUE}"; exit 1; }
    [[ -f ${POP_LABEL}.ini ]] || { echo "Missing gw-distribution initialization file"; exit 1; }

    mmms samples/${EVENT_SAMPLES}.csv.gz \
         ${POP_LABEL}.ini \
         ${FOLDER_NAME}/PDBpopulation${POP_VALUE}.csv.gz \
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

printf "\n\n[STATUS] Completed conversion PDB mmms, running multiPDB mmms now...\n"

for ENTRY in "${ALL_EVENTS[@]}"; do
  # Split into key and value parts
  EVENT_SAMPLES=${ENTRY%%|*}
  VALUE=${ENTRY#*|}
  for COMPONENT in $VALUE; do

  # Run individual mmms

  # Define population labels
  POP_LABEL="multiPDB_betaSplit_brokenG"
  SEED="--seed 7236"

  # EXTRA_EVENT_ARGS="--mass-column mass${COMPONENT}_source"
  EXTRA_EVENT_ARGS="--mass-column mass${COMPONENT}_source"
  POP_ARGS="$POP_MAX_ARG --mtov-column notch_lowmass_scale"

  LABEL="${EVENT_SAMPLES}+${POP_LABEL}+${POP_PARAM}+${POP_VALUE}+component${COMPONENT}"

  FOLDER_NAME="${PWD##*/}"

  cd .. || { echo "Failed to cd .."; exit 1; }

  # Run mmms for each POP_VALUE in parallel
  for POP_VALUE in "${POP_VALUES[@]}"; do
    LABEL="${EVENT_SAMPLES}+${POP_LABEL}+${POP_PARAM}+${POP_VALUE}+component${COMPONENT}"

    # echo "${FOLDER_NAME}/population${POP_VALUE}.csv.gz"
    echo "[STATUS] Running mmms for ${LABEL}..."

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


# Define events to use along with waveform type
ALL_EVENTS=(
  "GW190814_C01:IMRPhenomXPHMconst_jac|2"
)

printf "\n\n[STATUS] Completed conversion PDB mmms, running multiPDB mmms now...\n"

for ENTRY in "${ALL_EVENTS[@]}"; do
  # Split into key and value parts
  EVENT_SAMPLES=${ENTRY%%|*}
  VALUE=${ENTRY#*|}
  for COMPONENT in $VALUE; do

  # Run individual mmms

  # Define population labels
  POP_LABEL="constJAC_betaSplit_brokenG"
  POP_LABEL_INI="multiPDB_betaSplit_brokenG"
  SEED="--seed 7236"

  # EXTRA_EVENT_ARGS="--mass-column mass${COMPONENT}_source"
  EXTRA_EVENT_ARGS="--mass-column mass${COMPONENT}_source"
  POP_ARGS="$POP_MAX_ARG --mtov-column notch_lowmass_scale"

  LABEL="${EVENT_SAMPLES}+${POP_LABEL}+${POP_PARAM}+${POP_VALUE}+component${COMPONENT}"

  FOLDER_NAME="${PWD##*/}"

  cd .. || { echo "Failed to cd .."; exit 1; }

  # Run mmms for each POP_VALUE in parallel
  for POP_VALUE in "${POP_VALUES[@]}"; do
    LABEL="${EVENT_SAMPLES}+${POP_LABEL}+${POP_PARAM}+${POP_VALUE}+component${COMPONENT}"

    # echo "${FOLDER_NAME}/population${POP_VALUE}.csv.gz"

    echo "[STATUS] Running mmms for ${LABEL}..."

    # Assertions
    [[ -f samples/${EVENT_SAMPLES}.csv.gz ]] || { echo "Missing event samples file"; exit 1; }
    [[ -f ${FOLDER_NAME}/population${POP_VALUE}.csv.gz ]] || { echo "Missing pop samples file for POP_VALUE=${POP_VALUE}"; exit 1; }
    [[ -f ${POP_LABEL_INI}.ini ]] || { echo "Missing gw-distribution initialization file"; exit 1; }

    mmms samples/${EVENT_SAMPLES}.csv.gz \
         ${POP_LABEL_INI}.ini \
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


printf "\n\n[STATUS] Running with different PE now...\n"
ALL_EVENTS=(
  # "GW230529_Combined_PHM_highSpin|1"
  # "GW190425_C01:IMRPhenomPv2_NRTidal:HighSpin|1 2"
  "GW190814_C01:IMRPhenomDtestPE|2"
  # "GW190814_C01:MixedTEST|2"
  # "GW190814_posterior_samplesTEST|2"
  # "GW190917_C01:IMRPhenomXPHM|1 2"
  # "GW200105_C01:IMRPhenomXPHM|1 2"
  # "GW200115_C01:IMRPhenomNSBH:HighSpin|2"
)


for ENTRY in "${ALL_EVENTS[@]}"; do
  # Split into key and value parts
  EVENT_SAMPLES=${ENTRY%%|*}
  VALUE=${ENTRY#*|}
  for COMPONENT in $VALUE; do

  # Run individual mmms

  # Define population labels
  POP_LABEL="diffPE_betaSplit_brokenG"
  POP_LABEL_INI="multiPDB_betaSplit_brokenG"
  SEED="--seed 7236"

  # EXTRA_EVENT_ARGS="--mass-column mass${COMPONENT}_source"
  EXTRA_EVENT_ARGS="--mass-column mass${COMPONENT}_source"
  POP_ARGS="$POP_MAX_ARG --mtov-column notch_lowmass_scale"

  LABEL="${EVENT_SAMPLES}+${POP_LABEL}+${POP_PARAM}+${POP_VALUE}+component${COMPONENT}"

  FOLDER_NAME="${PWD##*/}"

  cd .. || { echo "Failed to cd .."; exit 1; }

  # Run mmms for each POP_VALUE in parallel
  for POP_VALUE in "${POP_VALUES[@]}"; do
    LABEL="${EVENT_SAMPLES}+${POP_LABEL}+${POP_PARAM}+${POP_VALUE}+component${COMPONENT}"

    # echo "${FOLDER_NAME}/population${POP_VALUE}.csv.gz"

    echo "[STATUS] Running mmms for ${LABEL}..."

    # Assertions
    [[ -f samples/${EVENT_SAMPLES}.csv.gz ]] || { echo "Missing event samples file"; exit 1; }
    [[ -f ${FOLDER_NAME}/population${POP_VALUE}.csv.gz ]] || { echo "Missing pop samples file for POP_VALUE=${POP_VALUE}"; exit 1; }
    [[ -f ${POP_LABEL_INI}.ini ]] || { echo "Missing gw-distribution initialization file"; exit 1; }

    mmms samples/${EVENT_SAMPLES}.csv.gz \
         ${POP_LABEL_INI}.ini \
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





