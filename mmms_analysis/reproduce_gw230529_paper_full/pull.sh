# NAME="S230529ay_EXP6_data0_1369419318-7460938_analysis_L1_merge_result.csv.gz"

# scp utkarsh.mali@ldas-grid.ligo.caltech.edu:/home/reed.essick/mmms-gw230529/etc/${NAME} ../samples/${NAME}

pwd="/home/utkarsh.mali/projects/mmms_test/mmms-gw230529/etc"
NAME="Prod-BBH-HighSpin1-HighSpin2.csv.gz"

scp utkarsh.mali@ldas-grid.ligo.caltech.edu:${pwd}/${NAME} ../samples/${NAME}
