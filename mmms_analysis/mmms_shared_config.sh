# shared_config.sh

# Common population arguments
POP_MAX_NUM_SAMPLES=1000
POP_MAX_ARG="--pop-max-num-samples $POP_MAX_NUM_SAMPLES"

# Event samples to use
EVENT_SAMPLES="GW190917_C01:IMRPhenomXPHM"

EVENT_ARGS=""
EVENT_ARGS="$EVENT_ARGS --event-max-num-samples 10000"
EVENT_ARGS="$EVENT_ARGS --prior-column logprior"
EVENT_ARGS="$EVENT_ARGS --prior-is-log"
EVENT_ARGS="$EVENT_ARGS --m1-column mass1_source"
EVENT_ARGS="$EVENT_ARGS --m2-column mass2_source"
EVENT_ARGS="$EVENT_ARGS --d-column luminosity_distance"
EVENT_ARGS="$EVENT_ARGS --q-range 0.0 1.0"
EVENT_ARGS="$EVENT_ARGS --m-range 0.0 20.0"
EVENT_ARGS="$EVENT_ARGS --mc-range 0.0 20.0"
EVENT_ARGS="$EVENT_ARGS --d-range 0.0 10000.0"