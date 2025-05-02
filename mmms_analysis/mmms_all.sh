#!/bin/bash

chmod +x mmms_pdb.sh mmms_pdb_eos.sh

# Run scripts in parallel
./mmms_pdb.sh &
./mmms_pdb_eos.sh &

# Wait for all to finish
wait

