#! /bin/bash

# Remove all files that end with .out, .err, or .csv.gz in the current directory
echo "[STATUS] Cleaning up old output files..."
rm -f *.out *.err *.csv.gz || echo "[STATUS] No old output files to clean up."
rm -f eos_population_mixtures/*.csv.gz || echo "[STATUS] No old eos_population_mixtures files to clean up."
echo "[STATUS] Old output files cleaned."

# Import shared config
# source ../mmms_shared_config.sh

# print current working directory
echo "[STATUS] Current working directory: $(pwd)"

source ../mmms_shared_config.sh
source ../mmms_eos_shared_config.sh

# Equation of state samples
EOS_NAME="LEC-2020"
EOS_SAMPLES="../samples/${EOS_NAME}"

# Fixing pop samples count for now
POP_MAX_NUM_SAMPLES=100
EOS_MAX_NUM_SAMPLES=500
POP_MAX_ARG="--pop-max-num-samples $POP_MAX_NUM_SAMPLES"


# Define events to use along with waveform type
ALL_EVENTS=(
  "GW230529_Combined_PHM_highSpin|1"
  "GW190425_C01:IMRPhenomPv2_NRTidal:HighSpin|1 2"
  "GW190814_C01:IMRPhenomXPHM|2"
  "GW190917_C01:IMRPhenomXPHM|1 2"
  "GW200105_C01:IMRPhenomXPHM|1 2"
  "GW200115_C01:IMRPhenomNSBH:HighSpin|2"
)
POP_PARAM="mu_costilt"
POP_VALUES=(-1.0 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# Running conversion script, you need to change the value of pop_param in the conversion script
for POP_VALUE in "${POP_VALUES[@]}"; do
  # THE MOST IMPORTANT CHANGING VARIABLE NEEDS TO BE THE FIRST ELEMENT IN LIST!
  echo "[STATUS] Running conversion for POP_VALUE=${POP_VALUE}..."
  python3 convert_pdbNG_betaSplit3_brokenG.py \
    --pop_param "mean_spin2_cos_polar_angle_spin2_polar_angle_1_mass2_source_0" "mean_spin2_cos_polar_angle_spin2_polar_angle_1_mass2_source_1" "mean_spin1_cos_polar_angle_spin1_polar_angle_1_mass1_source_0" "mean_spin1_cos_polar_angle_spin1_polar_angle_1_mass1_source_1" "stdv_spin2_cos_polar_angle_spin2_polar_angle_1_mass2_source_0" "stdv_spin1_cos_polar_angle_spin1_polar_angle_1_mass1_source_0" "stdv_spin1_cos_polar_angle_spin1_polar_angle_1_mass1_source_1" "stdv_spin2_cos_polar_angle_spin2_polar_angle_1_mass2_source_1"\
    --pop_value "$POP_VALUE" "$POP_VALUE" "$POP_VALUE" "$POP_VALUE" 0.5 0.5 0.5 0.5& 
done

wait

# Merge EOS and population
for POP_VALUE in "${POP_VALUES[@]}"; do
  echo "[STATUS] Merging EOS and Population for POP_VALUE=${POP_VALUE}..."
  POP="population${POP_VALUE}"
  OUT="eos_population_mixtures/${EOS_NAME}_population${POP_VALUE}"

  # Ensure the input files exist
  [[ -f ${POP}.csv.gz ]] || { echo "Missing population file: ${POP}.csv"; exit 1; }
  [[ -f ${EOS_SAMPLES}.csv.gz ]] || { echo "Missing EOS file: ${EOS_SAMPLES}.csv"; exit 1; }

  mmms-combine \
      --samples ${EOS_SAMPLES}.csv.gz ${EOS_MAX_NUM_SAMPLES} \
      --samples ${POP}.csv.gz ${POP_MAX_NUM_SAMPLES} \
      --outpath ${OUT}.csv.gz \
      ${SEED} \
      --verbose \
  || { echo "Failed to merge EOS and Population"; exit 1; } & 
done

wait

printf "\n\n[STATUS] Completed conversion scripts, running mmms now...\n"

for ENTRY in "${ALL_EVENTS[@]}"; do
  # Split into key and value parts
  EVENT_SAMPLES=${ENTRY%%|*}
  VALUE=${ENTRY#*|}
  for COMPONENT in $VALUE; do

  # Run individual mmms

  # Define population labels
  POP_LABEL="pdbNG_betaSplit3_brokenG"
  SEED="--seed 7236"

  # EXTRA_EVENT_ARGS="--mass-column mass${COMPONENT}_source"
  # POP_ARGS="$POP_MAX_ARG --mtov-column notch_lowmass_scale"
  EXTRA_EVENT_ARGS="--mass-column mass${COMPONENT}_source --spin-column spin${COMPONENT}_magnitude"
  EXTRA_POP_ARGS="--pop-weight-column ${EOS_WEIGHT}"

  LABEL="${EVENT_SAMPLES}+${POP_LABEL}+${POP_PARAM}+${POP_VALUE}+component${COMPONENT}"

  # Assertions
  [[ -f ../samples/${EVENT_SAMPLES}.csv.gz ]] || { echo "Missing event samples file"; exit 1; }
  [[ -f eos_population_mixtures/${EOS_NAME}_population${POP_VALUE}.csv.gz ]] || { echo "Missing pop samples file"; exit 1; }
  [[ -f ../${POP_LABEL}.ini ]] || { echo "Missing gw-distribution initialization file"; exit 1; }

  FOLDER_NAME="${PWD##*/}"

  cd .. || { echo "Failed to cd .."; exit 1; }

  # Run mmms for each POP_VALUE in parallel
  for POP_VALUE in "${POP_VALUES[@]}"; do
    echo "[STATUS] Running mmms for event ${EVENT_SAMPLES}, value ${POP_VALUE}, component ${COMPONENT}..."
    LABEL="${EVENT_SAMPLES}+${POP_LABEL}+${POP_PARAM}+${POP_VALUE}+component${COMPONENT}"

    # Assertions
    [[ -f samples/${EVENT_SAMPLES}.csv.gz ]] || { echo "Missing event samples file"; exit 1; }
    [[ -f ${FOLDER_NAME}/eos_population_mixtures/${EOS_NAME}_population${POP_VALUE}.csv.gz ]] || { echo "Missing pop samples file"; exit 1; }    
    [[ -f ${POP_LABEL}.ini ]] || { echo "Missing gw-distribution initialization file"; exit 1; }

    # mmms samples/${EVENT_SAMPLES}.csv.gz \
    #      ${POP_LABEL}.ini \
    #      ${FOLDER_NAME}/population${POP_VALUE}.csv.gz \
    #      ${EVENT_ARGS} \
    #      ${EXTRA_EVENT_ARGS} \
    #      ${POP_ARGS} \
    #      ${SEED} \
    #      1> ${FOLDER_NAME}/${LABEL}.out \
    #      2> ${FOLDER_NAME}/${LABEL}.err &
    mmms \
    samples/${EVENT_SAMPLES}.csv.gz \
    ${POP_LABEL}.ini \
    ${FOLDER_NAME}/eos_population_mixtures/${EOS_NAME}_population${POP_VALUE}.csv.gz \
    ${EVENT_ARGS} \
    ${EXTRA_EVENT_ARGS} \
    ${POP_ARGS} \
    ${EXTRA_POP_ARGS} \
    ${SEED} \
    1> ${FOLDER_NAME}/${LABEL}.out \
    2> ${FOLDER_NAME}/${LABEL}.err &
    wait 
  done

  cd "${FOLDER_NAME}" || { echo "Failed to cd back to original dir"; exit 1; }
  done
  wait
done

wait  # wait for all background jobs to finish

printf " \n[COMPLETED]\n "
