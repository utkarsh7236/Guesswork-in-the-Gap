#!/bin/bash

chmod +x ./*.sh

# Define list of event sample names
EVENT_SAMPLES_LIST=(
"GW230529_Combined_PHM_highSpin"
"GW230529_Combined_PHM_lowSecondarySpin"
"GW190425_C01:IMRPhenomPv2_NRTidal:HighSpin"
"GW190814_C01:IMRPhenomXPHM"
"GW190917_C01:IMRPhenomXPHM"
"GW200105_C01:IMRPhenomXPHM"
"GW200115_C01:IMRPhenomNSBH:HighSpin"
)
#EVENT_SAMPLES_LIST=("GW230529_Combined_PHM_lowSecondarySpin")

#EOS_WEIGHTS_LST=("logweight_PSR_GW" "logweight_PSR_GW_Xray")
EOS_WEIGHTS_LST=("logweight_PSR_GW_Xray")


for EVENT_SAMPLES in "${EVENT_SAMPLES_LIST[@]}"; do
  sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh

  echo "Running scripts for EVENT_SAMPLES=$EVENT_SAMPLES"
  ./mmms_pdbNG_betaSplit_brokenG.sh &
  ./mmms_pdbNG_betaSplit_brokenG_same_mbrk.sh &
  ./mmms_pdbNG_betaSplit_brokenG_sig_peak1_large.sh &
  ./mmms_pdbNG_betaSplit_brokenG_sig_peak1_test.sh &
  ./mmms_pdbNG_betaSplit_brokenG_tight_prior.sh &
  ./mmms_pdbNG_betaSplit3_brokenG.sh &
  ./mmms_pdbNG_betaSplitSmooth_brokenG.sh &
  ./mmms_pdbNG_betaSplit_singleG.sh &
  ./mmms_multiPDB_betaSplit_brokenG.sh &
  ./mmms_multiPDB_betaSplit_singleG.sh &
  ./mmms_multiPDB_betaSplit3_brokenG.sh &
  ./mmms_multiPDB_betaSplitSmooth_brokenG.sh &

#  for EOS_WEIGHT in "${EOS_WEIGHTS_LST[@]}"; do
#    sed -i '' "s/^EOS_WEIGHT=.*/EOS_WEIGHT=\"$EOS_WEIGHT\"/" mmms_eos_shared_config.sh
#    echo "    Running scripts for EOS_WEIGHT=$EOS_WEIGHT"
#    wait
#    ./mmms_eos_pdbNG_betaSplit_brokenG.sh &
#    ./mmms_eos_pdbNG_betaSplit_brokenG_same_mbrk.sh &
#    ./mmms_eos_pdbNG_betaSplit_brokenG_sig_peak1_large.sh &
#    ./mmms_eos_pdbNG_betaSplit_brokenG_sig_peak1_test.sh &
#    wait
#    ./mmms_eos_pdbNG_betaSplit_brokenG_tight_prior.sh &
#    ./mmms_eos_pdbNG_betaSplit3_brokenG.sh &
#    ./mmms_eos_pdbNG_betaSplitSmooth_brokenG.sh &
#    ./mmms_eos_pdbNG_betaSplit_singleG.sh &
#    wait
#    ./mmms_eos_multiPDB_betaSplit_brokenG.sh &
#    ./mmms_eos_multiPDB_betaSplit_singleG.sh &
#    ./mmms_eos_multiPDB_betaSplit3_brokenG.sh &
#    ./mmms_eos_multiPDB_betaSplitSmooth_brokenG.sh &
#    wait
#  done
done
# Wait for all to finish
wait

