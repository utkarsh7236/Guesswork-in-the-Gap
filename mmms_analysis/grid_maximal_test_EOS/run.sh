rm -f *.out *.err *.csv.gz || echo "[STATUS] No old output files to clean up."
rm -f eos_population_mixtures/*.csv.gz || echo "[STATUS] No old eos_population_mixtures files to clean up."

./run_A.sh
./run_B.sh