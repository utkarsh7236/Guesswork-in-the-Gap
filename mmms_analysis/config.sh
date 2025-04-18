#!/bin/bash

# a script that declares the sample sets we will process
# Reed Essick (reed.essick@gmail.com)

#-------------------------------------------------

# use a single source of EoS weights, just switch which columns we use to weigh the sample set
EOS_SAMPLES="LEC-2020"

# will iterate over which set of weights we use
EOS_WEIGHTS=""

#EOS_WEIGHTS="$EOS_WEIGHTS logweight_PSR"
EOS_WEIGHTS="$EOS_WEIGHTS logweight_PSR_GW"
EOS_WEIGHTS="$EOS_WEIGHTS logweight_PSR_GW_Xray"

#------------------------

PDB_SAMPLE_SETS=""

#PDB_SAMPLE_SETS="$PDB_SAMPLE_SETS Prelim-LVK-O3-230529-PDB"

#PDB_SAMPLE_SETS="$PDB_SAMPLE_SETS LVK-O3-230529-PDB"
#PDB_SAMPLE_SETS="$PDB_SAMPLE_SETS LVK-O3-230529-PDB-forced"

PDB_SAMPLE_SETS="$PDB_SAMPLE_SETS Updated-LVK-O3-sans-230529-PDB"
PDB_SAMPLE_SETS="$PDB_SAMPLE_SETS Updated-LVK-O3-sans-230529-PDB-forced"

PDB_SAMPLE_SETS="$PDB_SAMPLE_SETS Updated-LVK-O3-with-230529-PDB"
PDB_SAMPLE_SETS="$PDB_SAMPLE_SETS Updated-LVK-O3-with-230529-PDB-forced"

PDB_SAMPLE_SETS="$PDB_SAMPLE_SETS Farah2022-g-PDB"
PDB_SAMPLE_SETS="$PDB_SAMPLE_SETS Farah2022-h-PDB"
PDB_SAMPLE_SETS="$PDB_SAMPLE_SETS Farah2022-i-PDB"

#-----------

POP_SAMPLE_SETS=""

POP_SAMPLE_SETS="$POP_SAMPLE_SETS Default-PE"
POP_SAMPLE_SETS="$POP_SAMPLE_SETS $PDB_SAMPLE_SETS"

#------------------------

EVENT_SAMPLE_SETS=""

#EVENT_SAMPLE_SETS="$EVENT_SAMPLE_SETS Unknown-Prelim"

#EVENT_SAMPLE_SETS="$EVENT_SAMPLE_SETS BBH-HighSpin1-LowSpin2"
#EVENT_SAMPLE_SETS="$EVENT_SAMPLE_SETS BBH-HighSpin1-HighSpin2"

EVENT_SAMPLE_SETS="$EVENT_SAMPLE_SETS Prod-BBH-HighSpin1-HighSpin2"
#EVENT_SAMPLE_SETS="$EVENT_SAMPLE_SETS Prod-BBH-HighSpin1-LowSpin2"

#EVENT_SAMPLE_SETS="$EVENT_SAMPLE_SETS Prod-BBH-LowSpin1-LowSpin2" ### NOTE: do not use; this breaks the code

#------------------------

COMPONENTS=""

COMPONENTS="$COMPONENTS 1"
COMPONENTS="$COMPONENTS 2"
