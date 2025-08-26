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
EOS_MAX_NUM_SAMPLES=5000
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

POP_PARAM="beta_1"
POP_VALUES=(-0.5)   # Must be a float

# Running conversion script, you need to change the value of pop_param in the conversion script
for POP_VALUE in "${POP_VALUES[@]}"; do
  # THE MOST IMPORTANT CHANGING VARIABLE NEEDS TO BE THE FIRST ELEMENT IN LIST!
  echo "[STATUS] Running conversion for POP_VALUE=${POP_VALUE}..."
  python3 convert_multiPDB_betaSplit3_brokenG.py \
    --pop_param "$POP_PARAM" \
    --pop_value "$POP_VALUE" &
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