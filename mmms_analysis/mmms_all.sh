#!/bin/bash

chmod +x mmms_pdb.sh mmms_pdb_eos.sh

# Run scripts in parallel
./mmms_pdb.sh &
./mmms_pdb_eos.sh &
./mmms_pdbNG_betaSplit_brokenG.sh &
./mmms_multiPDB_betaSplit_brokenG.sh &

# Wait for all to finish
wait

