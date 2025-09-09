# shared_config.sh

# Common population arguments
POP_MAX_NUM_SAMPLES=100
POP_MAX_ARG="--pop-max-num-samples $POP_MAX_NUM_SAMPLES"

# EVENT_ARGS=""
# EVENT_ARGS="$EVENT_ARGS --event-max-num-samples 10000"
# EVENT_ARGS="$EVENT_ARGS --prior-column logprior"
# EVENT_ARGS="$EVENT_ARGS --prior-is-log"
# EVENT_ARGS="$EVENT_ARGS --m1-column mass1_source"
# EVENT_ARGS="$EVENT_ARGS --m2-column mass2_source"
# EVENT_ARGS="$EVENT_ARGS --d-column luminosity_distance"
# EVENT_ARGS="$EVENT_ARGS --q-range 0.0 1.0"
# EVENT_ARGS="$EVENT_ARGS --m-range 0.0 100.0"
# EVENT_ARGS="$EVENT_ARGS --mc-range 0.0 20.0"
# EVENT_ARGS="$EVENT_ARGS --d-range 0.0 10000.0"

# Makes no difference 
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



# EVENT_SAMPLES="GW230529_Combined_PHM_highSpin"
# EVENT_SAMPLES="GW230529_Combined_PHM_lowSecondarySpin" # Doesn't affect the results
# EVENT_SAMPLES="gw230529_highSpin" # Doesn't affect the results
# EVENT_SAMPLES="PROD2_S230529ay_PROD2_data0_1369419318-7460938_analysis_L1_merge_result" # Prod-BBH-HighSpin1-HighSpin2 samples 
EVENT_SAMPLES="Prod-BBH-HighSpin1-HighSpin2"
COMPONENT="1"

FOLDER_NAME="${PWD##*/}"

POP_FILE="population"
# POP_FILE="Updated-LVK-O3-sans-230529-PDB-forced"

# Define population labels
POP_LABEL="pdb_uniform"
# POP_LABEL="LVK-O3-230529-PDB"
SEED="--seed 123"

EXTRA_EVENT_ARGS="--mass-column mass${COMPONENT}_source"
POP_ARGS="$POP_MAX_ARG --mtov-column notch_lowmass_scale"

LABEL="${EVENT_SAMPLES}+${POP_LABEL}+${POP_PARAM}+${POP_VALUE}+component${COMPONENT}${POP_FILE}"

cd .. || { echo "Failed to cd .."; exit 1; }

[[ -f samples/${EVENT_SAMPLES}.csv.gz ]] || { echo "Missing event samples file"; exit 1; }
[[ -f ${FOLDER_NAME}/${POP_FILE}.csv.gz ]] || { echo "Missing pop samples file for POP_VALUE=${POP_VALUE}"; exit 1; }
[[ -f ${POP_LABEL}.ini ]] || { echo "Missing gw-distribution initialization file"; exit 1; }

mmms samples/${EVENT_SAMPLES}.csv.gz \
        ${POP_LABEL}.ini \
        ${FOLDER_NAME}/${POP_FILE}.csv.gz \
        ${EVENT_ARGS} \
        ${EXTRA_EVENT_ARGS} \
        ${POP_ARGS} \
        ${SEED} \
        --Verbose \
        1> ${FOLDER_NAME}/${LABEL}.out \
        2> ${FOLDER_NAME}/${LABEL}.err & 

#mmms-plot samples/${EVENT_SAMPLES}.csv.gz \
#            ${POP_LABEL}.ini \
#            ${FOLDER_NAME}/${POP_FILE}.csv.gz \
#            ${EVENT_ARGS} \
#            ${EXTRA_EVENT_ARGS} \
#            ${POP_ARGS} \
#            ${SEED} \
#    --output-dir ${FOLDER_NAME}/figures \
#    --tag ${LABEL} \
#    --figtype png --dpi 300

cd "${FOLDER_NAME}" || { echo "Failed to cd back to original dir"; exit 1; }

wait