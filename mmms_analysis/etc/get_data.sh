# gather the required input files for the analysis
# Reed Essick (reed.essick@gmail.com)

#-------------------------------------------------

# grab EoS posteriors

EOSS=""

#------------------------

# the file used in O3 NSBH detection paper
EOS="LEC-mtov-rtov"
echo $EOS

# register this EoS
EOSS="$EOSS $EOS"

#------------------------

# construct a more complete dataset so we can select which astro logweights to include
EOS="LEC-2020"
echo $EOS

scp utkarsh.mali@ldas-grid.ligo-wa.caltech.edu:/home/reed.essick/BNS_tides/gpr-eos-stacking/data/EOSeights_AGN-with-pressures.csv ./${EOS}-full.csv \
|| exit 1

# register this EoS
EOSS="$EOSS $EOS"

#-------------------------------------------------
POPS=""

#------------------------
# register this population to be combined with EoS posterior(s)
POPS="$POPS Default-PE"

#------------------------

POP="Prelim-LVK-O3-230529-PDB"
echo $POP

scp utkarsh.mali@ldas-grid.ligo.caltech.edu:/home/amanda.farah/projects/O4/during_run_pop_anlyses/with_S230529ay/result/O1O2O3all_mass_h_iid_mag_iid_tilt_powerlaw_redshift_result.json ./${POP}.json \
|| exit 1

# grab the classifications to go along with this Prelim analysis
scp utkarsh.mali@ldas-grid.ligo.caltech.edu:/home/amanda.farah/projects/O4/during_run_pop_anlyses/with_S230529ay/result/MatterMattersPairing.json ./${POP}-classifications.json \
|| exit 1

# register this population
POPS="$POPS $POP"

#-----------

POP="LVK-O3-230529-PDB"
echo $POP

# grab samples from CIT
scp utkarsh.mali@ldas-grid.ligo.caltech.edu:/home/amanda.farah/projects/O4/during_run_pop_anlyses/with_S230529ay/result/EXP0_morelive_mass_h_iid_mag_iid_tilt_powerlaw_redshift_result.json ./${POP}.json \
|| exit 1

# register this population to be combined with EoS posterior(s)
POPS="$POPS $POP"

#-----------

# register population
POPS="$POPS ${POP}-forced"

#------------------------

POP="Updated-LVK-O3-with-230529-PDB"
echo $POP

scp utkarsh.mali@ldas-grid.ligo.caltech.edu:/home/michael.zevin/projects/o4/gw230529/analysis_results/populations/PowerlawDipBreak/production/with_230529_mass_h_result.json ./${POP}.json \
|| exit 1

# register this population to be combined with EoS posterior(s)
POPS="$POPS $POP"

# register population
POPS="$POPS ${POP}-forced"

#-----------

POP="Updated-LVK-O3-sans-230529-PDB"
echo $POP

scp utkarsh.mali@ldas-grid.ligo.caltech.edu:/home/michael.zevin/projects/o4/gw230529/analysis_results/populations/PowerlawDipBreak/production/no_230529_mass_h_result.json ./${POP}.json \
|| exit 1

# register this population to be combined with EoS posterior(s)
POPS="$POPS $POP"

# register population
POPS="$POPS ${POP}-forced"

#------------------------

### Farah+2022 PDB models

for MODEL in "g" "h" "i"
do

    POP="Farah2022-${MODEL}-PDB"
    echo $POP

    # grab samples from CIT
    scp utkarsh.mali@ldas-grid.ligo.caltech.edu:/home/amanda.farah/projects/O3/population_runs/chips_dip/binned_pairing/mbreak_is_gammalow/result/o1o2o3a_mass_${MODEL}_iid_mag_iid_tilt_powerlaw_redshift_result.json ./${POP}.json \
    || exit 1

    # register this population to be combined with EoS posterior(s)
    POPS="$POPS $POP"

done

#------------------------

#-------------------------------------------------

echo "grabbing single-event posterior samples"

#------------------------

# BBH waveform : Exp6 (wide spin1 prior, narrow spin2 prior)
SAMP="BBH-HighSpin1-LowSpin2"
echo $SAMP

scp utkarsh.mali@ldas-grid.ligo.caltech.edu:/home/pe.o4/O4a/S230529ay/Exp6/bilby/final_result/S230529ay_EXP6_data0_1369419318-7460938_analysis_L1_merge_result.hdf5 ./${SAMP}.hdf5 \
|| exit 1

#------------------------

# BBH waveform: Prod2 (wide spin1 and spin2 priors)
SAMP="BBH-HighSpin1-HighSpin2"
echo $SAMP

scp utkarsh.mali@ldas-grid.ligo.caltech.edu:/home/geraint.pratten/public_html/O4/S230529ay/production/summarypages/PROD2/samples/PROD2_S230529ay_PROD2_data0_1369419318-7460938_analysis_L1_merge_result.hdf5 ./${SAMP}.hdf5 \
|| exit 1

#------------------------

# unknown waveform and priors; what Amanda Farah used in her preliminary PDB fit
SAMP="Unknown-Prelim"
echo $SAMP

scp utkarsh.mali@ldas-grid.ligo.caltech.edu:/home/amanda.farah/projects/O4/during_run_pop_anlyses/new_evs/S230529ay_online.h5 ./${SAMP}.hdf5 \
|| exit 1

#-------------------------------------------------
#
# PRODUCTION PE
#
#-------------------------------------------------

SAMP="Prod-BBH-HighSpin1-HighSpin2"
echo $SAMP

scp utkarsh.mali@ldas-grid.ligo.caltech.edu:/home/michael.zevin/projects/o4/gw230529/analysis_results/parameter_estimation/CombinedPHMHighSpin/posterior_samples.h5 ./${SAMP}.hdf5 \
|| exit 1

SAMP="Prod-BBH-HighSpin1-LowSpin2"
echo $SAMP

scp utkarsh.mali@ldas-grid.ligo.caltech.edu:/home/michael.zevin/projects/o4/gw230529/analysis_results/parameter_estimation/CombinedPHMLowSpin/posterior_samples.h5 ./${SAMP}.hdf5 \
|| exit 1

#------------------------

SAMP="Prod-BBH-LowSpin1-LowSpin2"
echo $SAMP

scp utkarsh.mali@ldas-grid.ligo.caltech.edu:/home/michael.zevin/projects/o4/gw230529/analysis_results/parameter_estimation/IMRPhenomXPHM/production/posterior_samples_low_spin_both.h5 ./${SAMP}.hdf5 \
|| exit 1
