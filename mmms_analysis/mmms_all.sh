#!/bin/bash

chmod +x ./*.sh

# Define list of event sample names
#EVENT_SAMPLES_LIST=("GW230529_Combined_PHM_highSpin", "Combined_PHM_lowSecondarySpin")
EVENT_SAMPLES_LIST=("Combined_PHM_lowSecondarySpin")

for EVENT_SAMPLES in "${EVENT_SAMPLES_LIST[@]}"; do
  sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh

  echo "Running scripts for EVENT_SAMPLES=$EVENT_SAMPLES"
  # Run scripts in parallel
  #./mmms_pdb.sh &
  #./mmms_pdb_eos.sh &
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
done
# Wait for all to finish
wait

