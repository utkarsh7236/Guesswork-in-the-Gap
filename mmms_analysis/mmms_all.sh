#!/bin/bash

chmod +x ./*.sh

# Run scripts in parallel
./mmms_pdb.sh &
./mmms_pdb_eos.sh &
./mmms_pdbNG_betaSplit_brokenG.sh &
./mmms_pdbNG_betaSplit_brokenG_same_mbrk.sh &
./mmms_pdbNG_betaSplit_brokenG_sig_peak1_large.sh &
./mmms_pdbNG_betaSplit_brokenG_sig_peak1_test.sh &
./mmms_pdbNG_betaSplit_brokenG_tight_prior.sh &
./mmms_pdbNG_betaSplit3_brokenG.sh &
./mmms_pdbNG_betaSplitSmooth_brokenG.sh &
./mmms_mmms_pdbNG_betaSplit_singleG.sh &
./mmms_multiPDB_betaSplit_brokenG.sh &
./mmms_multiPDB_betaSplit_singleG.sh &
./mmms_multiPDB_betaSplit3_brokenG.sh &
./mmms_multiPDB_betaSplitSmooth_brokenG.sh &

# Wait for all to finish
wait

