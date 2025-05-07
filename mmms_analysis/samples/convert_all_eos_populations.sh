#cd ../conversion_scripts || { echo "Failed to change directory"; exit 1; }
#./convert_all.sh
#cd ../samples || { echo "Failed to change directory"; exit 1; }

# Assert EOS scripts exist
[[ -f LEC-2020-full.csv ]] || { echo "Missing an EOS file"; exit 1; }
[[ -f LEC-mtov-rtov.csv ]] || { echo "Missing an EOS file"; exit 1; }

# Combine EOS Scripts
./lec-2020-combine || { echo "Failed to combine LEC-2020-full.csv and LEC-mtov-rtov.csv"; exit 1; }

EOS="LEC-2020"
POP_FOLDER="pdbNG_betaSplit_brokenG"
POP="../conversion_scripts/${POP_FOLDER}/population"
OUT="eos_population_mixtures/${POP_FOLDER}_${EOS}"
SIZE=1000

## Merge EOS and Population
##TODO: Figure out how to make this work for one case, before you try to make it work for all
mmms-combine \
    --samples ${EOS}.csv ${SIZE} \
    --samples ${POP}.csv ${SIZE} \
    --outpath ${OUT}.csv.gz \
    --seed 123 \
    --verbose \
|| { echo "Failed to merge EOS and Population"; exit 1; }



