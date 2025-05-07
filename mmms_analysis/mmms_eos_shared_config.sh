# shared_config.sh
 source mmms_shared_config.sh

POP_ARGS="$POP_MAX_ARG"
POP_ARGS="$POP_ARGS --mtov-column Mmax"
POP_ARGS="$POP_ARGS --rtov-column Rmax"
POP_ARGS="$POP_ARGS --pop-weight-is-log"

EOS_WEIGHT=""
#EOS_WEIGHT="$EOS_WEIGHTS logweight_PSR_GW"
#EOS_WEIGHT="$EOS_WEIGHTS logweight_PSR_GW_Xray"
