#!/bin/bash

# Import shared config
source ../mmms_shared_config.sh

# Remove all files that end with .out, .err, or .csv.gz in the current directory
echo "[STATUS] Cleaning up old output files..."
# rm -f *.out *.err *.csv.gz
echo "[STATUS] Old output files cleaned."

# Fixing pop samples count for now
POP_MAX_NUM_SAMPLES=1000
POP_MAX_ARG="--pop-max-num-samples $POP_MAX_NUM_SAMPLES"

ALL_EVENTS=(
  "GW230529_Combined_PHM_highSpin|1"
  "GW190425_C01:IMRPhenomPv2_NRTidal:HighSpin|1"
  "GW190814_C01:IMRPhenomXPHM|2"
)

POP_PARAM="beta_1"
POP_VALUES=(-5.0)   # Must be a float

MU_TILT_VALUE=1

# Running conversion script, you need to change the value of pop_param in the conversion script
for POP_VALUE in "${POP_VALUES[@]}"; do
  # THE MOST IMPORTANT CHANGING VARIABLE NEEDS TO BE THE FIRST ELEMENT IN LIST!
  echo "[STATUS] Running conversion for POP_VALUE=${POP_VALUE}..."
  python3 convert_pdbNG_betaSplit3_brokenG.py \
    --pop_param "$POP_PARAM" "mean_spin2_cos_polar_angle_spin2_polar_angle_1_mass2_source_0" "mean_spin2_cos_polar_angle_spin2_polar_angle_1_mass2_source_1" "mean_spin1_cos_polar_angle_spin1_polar_angle_1_mass1_source_0" "mean_spin1_cos_polar_angle_spin1_polar_angle_1_mass1_source_1" "stdv_spin2_cos_polar_angle_spin2_polar_angle_1_mass2_source_0" "stdv_spin1_cos_polar_angle_spin1_polar_angle_1_mass1_source_0" "stdv_spin1_cos_polar_angle_spin1_polar_angle_1_mass1_source_1" "stdv_spin2_cos_polar_angle_spin2_polar_angle_1_mass2_source_1"\
    --pop_value "$POP_VALUE" "$MU_TILT_VALUE" "$MU_TILT_VALUE" "$MU_TILT_VALUE" "$MU_TILT_VALUE" 0.5 0.5 0.5 0.5& 
done

# Wait for all backgrounded conversions to finish
wait

printf "\n\n[STATUS] Completed conversion scripts, running mmax-model-selection now...\n"

for ENTRY in "${ALL_EVENTS[@]}"; do
  # Split into key and value parts
  EVENT_SAMPLES=${ENTRY%%|*}
  VALUE=${ENTRY#*|}
  for COMPONENT in $VALUE; do
  # Define population labels
  POP_LABEL="pdbNG_betaSplit3_brokenG"
  SEED="--seed 7236"

  EXTRA_EVENT_ARGS="--mass-column mass${COMPONENT}_source"
  POP_ARGS="$POP_MAX_ARG --mtov-column notch_lowmass_scale"

  LABEL="${EVENT_SAMPLES}+${POP_LABEL}+${POP_PARAM}+${POP_VALUE}+component${COMPONENT}"

  echo "[STATUS] Running mmms for ${LABEL}..."

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