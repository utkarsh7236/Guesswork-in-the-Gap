#!/bin/bash

# rm -f *.out *.err *.csv.gz || echo "[STATUS] No old output files to clean up."
# rm -f eos_population_mixtures/*.csv.gz || echo "[STATUS] No old eos_population_mixtures files to clean up."

./run_A.sh & 
./run_B.sh & 

# JUST RUN GW190814 CASE TO CHECK INTUITION
# ./run_C.sh &
# ./run_D.sh &

wait 
printf " \n[COMPLETED]\n "