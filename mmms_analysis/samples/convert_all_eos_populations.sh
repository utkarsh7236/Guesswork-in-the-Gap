cd ../conversion_scripts || { echo "Failed to change directory"; exit 1; }
chmod +x ./*.sh
./convert_all.sh
cd ../samples || { echo "Failed to change directory"; exit 1; }

# Assert EOS scripts exist
[[ -f LEC-2020-full.csv ]] || { echo "Missing an EOS file"; exit 1; }
[[ -f LEC-mtov-rtov.csv ]] || { echo "Missing an EOS file"; exit 1; }

# Combine EOS Scripts
./lec-2020-combine || { echo "Failed to combine LEC-2020-full.csv and LEC-mtov-rtov.csv"; exit 1; }

POP_FOLDER_LIST=(
  "multiPDB_betaSplit3_brokenG"
  "multiPDB_betaSplit_brokenG"
  "multiPDB_betaSplit_singleG"
  "multiPDB_betaSplitSmooth_brokenG"
  "pdbNG_betaSplit3_brokenG"
  "pdbNG_betaSplit_brokenG"
  "pdbNG_betaSplit_brokenG_same_mbrk"
  "pdbNG_betaSplit_brokenG_sig_peak1_large"
  "pdbNG_betaSplit_brokenG_sig_peak1_test"
  "pdbNG_betaSplit_singleG"
  "pdbNG_betaSplit_brokenG_tight_prior"
  "pdbNG_betaSplitSmooth_brokenG"
  "multiPDB_betaSplit_brokenG_G"
  "multiPDB_betaSplit_brokenG_I"
  "multiPDB_betaSplit_brokenG_K"
  "multiPDB_betaSplit_brokenG_M"
  "multiPDB_betaSplit_brokenG_N"
  "multiPDB_betaSplit_brokenG_P"
  "pdbNG_betaSplit_brokenG_A"
  "pdbNG_betaSplit_brokenG_B"
  "pdbNG_betaSplit_brokenG_C"
  "pdbNG_betaSplit_brokenG_D"
  "pdbNG_betaSplit_brokenG_E"
  "pdbNG_betaSplit_brokenG_F"
  "pdbNG_betaSplit_brokenG_H"
  "pdbNG_betaSplit_brokenG_J"
  "pdbNG_betaSplit_brokenG_L"
  "pdbNG_betaSplit_brokenG_O"
  "pdbNG_betaSplit_brokenG_Q"
)
EOS="LEC-2020"
SIZE=1000

## Merge EOS and Population
for POP_FOLDER in "${POP_FOLDER_LIST[@]}"; do
  POP="../conversion_scripts/${POP_FOLDER}/population"
  OUT="eos_population_mixtures/${POP_FOLDER}_${EOS}"
  mmms-combine \
      --samples ${EOS}.csv ${SIZE} \
      --samples ${POP}.csv ${SIZE} \
      --outpath ${OUT}.csv.gz \
      --seed 123 \
      --verbose \
  || { echo "Failed to merge EOS and Population"; exit 1; }
done


