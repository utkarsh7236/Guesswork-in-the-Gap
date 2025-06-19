##!/bin/bash
## Currently 239 scripts to run

# GW230529 (primary)
# GW190425 (primary and secondary)
# GW190814 (secondary)
# GW190917 (primary and secondary)
# GW200105 (primary and secondary)
# GW200115 (secondary)
ALL_EVENTS=(
  "GW230529_Combined_PHM_highSpin|1"
  "GW230529_Combined_PHM_lowSecondarySpin|1"
  "GW190425_C01:IMRPhenomPv2_NRTidal:HighSpin|1 2"
  "GW190814_C01:IMRPhenomXPHM|2"
  "GW190917_C01:IMRPhenomXPHM|1 2"
  "GW200105_C01:IMRPhenomXPHM|1 2"
  "GW200115_C01:IMRPhenomNSBH:HighSpin|2"
)

#6. All events compare default multiPDB vs default pdbNG
POP_LABEL_SUFFIX=""
POP_LABEL="pdbNG_betaSplit_brokenG"
for ENTRY in "${ALL_EVENTS[@]}"; do
  # Split into key and value parts
  EVENT_SAMPLES=${ENTRY%%|*}
  VALUE=${ENTRY#*|}
  for COMPONENT in $VALUE; do
    sed -i '' "s|^EVENT_SAMPLES=.*|EVENT_SAMPLES=\"$EVENT_SAMPLES\"|" mmms_shared_config.sh
    sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
    sed -i '' "s|^COMPONENT=.*|COMPONENT=\"$COMPONENT\"|"      mmms_tasks.sh
    sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
    ./mmms_tasks.sh
  done
done

#POP_LABEL_SUFFIX=""
#POP_LABEL="multiPDB_betaSplit_brokenG"
#for ENTRY in "${ALL_EVENTS[@]}"; do
#  # Split into key and value parts
#  EVENT_SAMPLES=${ENTRY%%|*}
#  VALUE=${ENTRY#*|}
#  for COMPONENT in $VALUE; do
#    sed -i '' "s|^EVENT_SAMPLES=.*|EVENT_SAMPLES=\"$EVENT_SAMPLES\"|" mmms_shared_config.sh
#    sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#    sed -i '' "s|^COMPONENT=.*|COMPONENT=\"$COMPONENT\"|"      mmms_tasks.sh
#    sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#    ./mmms_tasks.sh
#  done
#done
#
#1. GW230529 (primary) with varying gamma_low, eta_low, and all other params fixed to MAP
#    - Task A
EVENT_SAMPLES="GW230529_Combined_PHM_highSpin"
POP_LABEL="pdbNG_betaSplit_brokenG"
POP_LABEL_SUFFIX="_A"
COMPONENT="1"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

EVENT_SAMPLES="GW230529_Combined_PHM_lowSecondarySpin"
POP_LABEL="pdbNG_betaSplit_brokenG"
POP_LABEL_SUFFIX="_A"
COMPONENT="1"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

##2. GW190814 (primary and secondary) with varying gamma_low, eta_low, and all other params fixed to MAP
##    - Task A
#EVENT_SAMPLES="GW190814_C01:IMRPhenomXPHM"
#POP_LABEL="pdbNG_betaSplit_brokenG"
#POP_LABEL_SUFFIX="_A"
#COMPONENT="2"
#sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
#sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
#sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#./mmms_tasks.sh &
#wait
#
##3. GW190425 (primary and secondary) with varying gamma_low, eta_low, and all other params fixed to MAP
##    - Task A
#EVENT_SAMPLES="GW190425_C01:IMRPhenomPv2_NRTidal:HighSpin"
#POP_LABEL="pdbNG_betaSplit_brokenG"
#POP_LABEL_SUFFIX="_A"
#COMPONENT="1"
#sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
#sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
#sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#./mmms_tasks.sh &
#wait
#
#EVENT_SAMPLES="GW190425_C01:IMRPhenomPv2_NRTidal:HighSpin"
#POP_LABEL="pdbNG_betaSplit_brokenG"
#POP_LABEL_SUFFIX="_A"
#COMPONENT="2"
#sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
#sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
#sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#./mmms_tasks.sh &
#wait
#
#4. GW230529 (primary) with fixed gamma_high, eta_high and with with fixed gamma_high, eta_high and all other params fixed to MAP
#    - Task B
#    - Task T
EVENT_SAMPLES="GW230529_Combined_PHM_highSpin"
POP_LABEL="pdbNG_betaSplit_brokenG"
POP_LABEL_SUFFIX="_B"
COMPONENT="1"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

EVENT_SAMPLES="GW230529_Combined_PHM_highSpin"
POP_LABEL="pdbNG_betaSplit_brokenG"
POP_LABEL_SUFFIX="_T"
COMPONENT="1"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

EVENT_SAMPLES="GW230529_Combined_PHM_lowSecondarySpin"
POP_LABEL="pdbNG_betaSplit_brokenG"
POP_LABEL_SUFFIX="_B"
COMPONENT="1"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

EVENT_SAMPLES="GW230529_Combined_PHM_lowSecondarySpin"
POP_LABEL="pdbNG_betaSplit_brokenG"
POP_LABEL_SUFFIX="_T"
COMPONENT="1"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait


##5. GW200115 (primary) with fixed gamma_high, eta_high and with with fixed gamma_high, eta_high and all other params fixed to MAP
##    - Task B
##    - Task T
#EVENT_SAMPLES="GW200115_C01:IMRPhenomNSBH:HighSpin"
#POP_LABEL="pdbNG_betaSplit_brokenG"
#POP_LABEL_SUFFIX="_B"
#COMPONENT="2"
#sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
#sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
#sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#./mmms_tasks.sh &
#wait
#
#EVENT_SAMPLES="GW200115_C01:IMRPhenomNSBH:HighSpin"
#POP_LABEL="pdbNG_betaSplit_brokenG"
#POP_LABEL_SUFFIX="_T"
#COMPONENT="2"
#sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
#sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
#sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#./mmms_tasks.sh &
#wait
#
##7. All events with default pdbNG but only sig_peak_NS < median
##    - Task C
#POP_LABEL_SUFFIX="_C"
#POP_LABEL="pdbNG_betaSplit_brokenG"
#for ENTRY in "${ALL_EVENTS[@]}"; do
#  # Split into key and value parts
#  EVENT_SAMPLES=${ENTRY%%|*}
#  VALUE=${ENTRY#*|}
#  for COMPONENT in $VALUE; do
#    sed -i '' "s|^EVENT_SAMPLES=.*|EVENT_SAMPLES=\"$EVENT_SAMPLES\"|" mmms_shared_config.sh
#    sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#    sed -i '' "s|^COMPONENT=.*|COMPONENT=\"$COMPONENT\"|"      mmms_tasks.sh
#    sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#    ./mmms_tasks.sh
#  done
#done
#
#
#8. GW230529 (primary) with fixed mu_peak_NS to Galactic BNS rate
#    - Task D and E
EVENT_SAMPLES="GW230529_Combined_PHM_highSpin"
POP_LABEL="pdbNG_betaSplit_brokenG"
POP_LABEL_SUFFIX="_D"
COMPONENT="1"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

EVENT_SAMPLES="GW230529_Combined_PHM_lowSecondarySpin"
POP_LABEL="pdbNG_betaSplit_brokenG"
POP_LABEL_SUFFIX="_D"
COMPONENT="1"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

EVENT_SAMPLES="GW230529_Combined_PHM_highSpin"
POP_LABEL="pdbNG_betaSplit_brokenG"
POP_LABEL_SUFFIX="_E"
COMPONENT="1"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

EVENT_SAMPLES="GW230529_Combined_PHM_lowSecondarySpin"
POP_LABEL="pdbNG_betaSplit_brokenG"
POP_LABEL_SUFFIX="_E"
COMPONENT="1"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait


##9. GW190814 (secondary) with fixed mu_peak_NS to Galactic BNS rate
##    - Task D and E
#EVENT_SAMPLES="GW190814_C01:IMRPhenomXPHM"
#POP_LABEL="pdbNG_betaSplit_brokenG"
#POP_LABEL_SUFFIX="_D"
#COMPONENT="2"
#sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
#sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
#sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#./mmms_tasks.sh &
#wait
#
#EVENT_SAMPLES="GW190814_C01:IMRPhenomXPHM"
#POP_LABEL="pdbNG_betaSplit_brokenG"
#POP_LABEL_SUFFIX="_E"
#COMPONENT="2"
#sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
#sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
#sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#./mmms_tasks.sh &
#wait
#
#
##10. GW200115 (primary) with fixed mu_peak_NS to Galactic BNS rate
##    - Task D and E
#EVENT_SAMPLES="GW200115_C01:IMRPhenomNSBH:HighSpin"
#POP_LABEL="pdbNG_betaSplit_brokenG"
#POP_LABEL_SUFFIX="_D"
#COMPONENT="2"
#sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
#sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
#sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#./mmms_tasks.sh &
#wait
#
#EVENT_SAMPLES="GW200115_C01:IMRPhenomNSBH:HighSpin"
#POP_LABEL="pdbNG_betaSplit_brokenG"
#POP_LABEL_SUFFIX="_E"
#COMPONENT="2"
#sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
#sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
#sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#./mmms_tasks.sh &
#wait
#
##11. GW190814 (secondary) with fixed beta_1 = beta_2 = 0
##    - Task F
##    - Task G
##    - Task R
##    - Task S
#EVENT_SAMPLES="GW190814_C01:IMRPhenomXPHM"
#POP_LABEL="pdbNG_betaSplit_brokenG"
#POP_LABEL_SUFFIX="_F"
#COMPONENT="2"
#sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
#sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
#sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#./mmms_tasks.sh &
#wait
#
#EVENT_SAMPLES="GW190814_C01:IMRPhenomXPHM"
#POP_LABEL="multiPDB_betaSplit_brokenG"
#POP_LABEL_SUFFIX="_G"
#COMPONENT="2"
#sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
#sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
#sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#./mmms_tasks.sh &
#wait
#
#EVENT_SAMPLES="GW190814_C01:IMRPhenomXPHM"
#POP_LABEL="pdbNG_betaSplit_brokenG"
#POP_LABEL_SUFFIX="_R"
#COMPONENT="2"
#sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
#sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
#sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#./mmms_tasks.sh &
#wait
#
#EVENT_SAMPLES="GW190814_C01:IMRPhenomXPHM"
#POP_LABEL="multiPDB_betaSplit_brokenG"
#POP_LABEL_SUFFIX="_S"
#COMPONENT="2"
#sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
#sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
#sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#./mmms_tasks.sh &
#wait
#
#12. GW230529 (primary) with fixed beta_1 = beta_2 = 0
#    - Task F
#    - Task G
#    - Task R
#    - Task S
EVENT_SAMPLES="GW230529_Combined_PHM_highSpin"
POP_LABEL="pdbNG_betaSplit_brokenG"
POP_LABEL_SUFFIX="_F"
COMPONENT="1"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

EVENT_SAMPLES="GW230529_Combined_PHM_highSpin"
POP_LABEL="multiPDB_betaSplit_brokenG"
POP_LABEL_SUFFIX="_G"
COMPONENT="1"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

EVENT_SAMPLES="GW230529_Combined_PHM_highSpin"
POP_LABEL="pdbNG_betaSplit_brokenG"
POP_LABEL_SUFFIX="_R"
COMPONENT="1"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

EVENT_SAMPLES="GW230529_Combined_PHM_highSpin"
POP_LABEL="multiPDB_betaSplit_brokenG"
POP_LABEL_SUFFIX="_S"
COMPONENT="1"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

EVENT_SAMPLES="GW230529_Combined_PHM_lowSecondarySpin"
POP_LABEL="pdbNG_betaSplit_brokenG"
POP_LABEL_SUFFIX="_F"
COMPONENT="1"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

EVENT_SAMPLES="GW230529_Combined_PHM_lowSecondarySpin"
POP_LABEL="multiPDB_betaSplit_brokenG"
POP_LABEL_SUFFIX="_G"
COMPONENT="1"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

EVENT_SAMPLES="GW230529_Combined_PHM_lowSecondarySpin"
POP_LABEL="pdbNG_betaSplit_brokenG"
POP_LABEL_SUFFIX="_R"
COMPONENT="1"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

EVENT_SAMPLES="GW230529_Combined_PHM_lowSecondarySpin"
POP_LABEL="multiPDB_betaSplit_brokenG"
POP_LABEL_SUFFIX="_S"
COMPONENT="1"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

##13. GW190814 (secondary) with fixed beta_1 = beta_2 = 4
##    - Task H
##    - Task I
#EVENT_SAMPLES="GW190814_C01:IMRPhenomXPHM"
#POP_LABEL="pdbNG_betaSplit_brokenG"
#POP_LABEL_SUFFIX="_H"
#COMPONENT="2"
#sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
#sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
#sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#./mmms_tasks.sh &
#wait
#
#EVENT_SAMPLES="GW190814_C01:IMRPhenomXPHM"
#POP_LABEL="multiPDB_betaSplit_brokenG"
#POP_LABEL_SUFFIX="_I"
#COMPONENT="2"
#sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
#sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
#sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#./mmms_tasks.sh &
#wait
#
#
#14. GW230529 (primary) with fixed beta_1 = beta_2 = 4
#    - Task H
#    - Task I
EVENT_SAMPLES="GW230529_Combined_PHM_highSpin"
POP_LABEL="pdbNG_betaSplit_brokenG"
POP_LABEL_SUFFIX="_H"
COMPONENT="1"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

EVENT_SAMPLES="GW230529_Combined_PHM_highSpin"
POP_LABEL="multiPDB_betaSplit_brokenG"
POP_LABEL_SUFFIX="_I"
COMPONENT="1"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

EVENT_SAMPLES="GW230529_Combined_PHM_lowSecondarySpin"
POP_LABEL="pdbNG_betaSplit_brokenG"
POP_LABEL_SUFFIX="_H"
COMPONENT="1"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

EVENT_SAMPLES="GW230529_Combined_PHM_lowSecondarySpin"
POP_LABEL="multiPDB_betaSplit_brokenG"
POP_LABEL_SUFFIX="_I"
COMPONENT="1"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

#15. All events with smooth pairing function for multiPDB
POP_LABEL_SUFFIX=""
POP_LABEL="pdbNG_betaSplitSmooth_brokenG"
for ENTRY in "${ALL_EVENTS[@]}"; do
  # Split into key and value parts
  EVENT_SAMPLES=${ENTRY%%|*}
  VALUE=${ENTRY#*|}
  for COMPONENT in $VALUE; do
    sed -i '' "s|^EVENT_SAMPLES=.*|EVENT_SAMPLES=\"$EVENT_SAMPLES\"|" mmms_shared_config.sh
    sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
    sed -i '' "s|^COMPONENT=.*|COMPONENT=\"$COMPONENT\"|"      mmms_tasks.sh
    sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
    ./mmms_tasks.sh
  done
done
#
##16. All events with smooth pairing function for pdbNG
#POP_LABEL_SUFFIX=""
#POP_LABEL="multiPDB_betaSplitSmooth_brokenG"
#for ENTRY in "${ALL_EVENTS[@]}"; do
#  # Split into key and value parts
#  EVENT_SAMPLES=${ENTRY%%|*}
#  VALUE=${ENTRY#*|}
#  for COMPONENT in $VALUE; do
#    sed -i '' "s|^EVENT_SAMPLES=.*|EVENT_SAMPLES=\"$EVENT_SAMPLES\"|" mmms_shared_config.sh
#    sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#    sed -i '' "s|^COMPONENT=.*|COMPONENT=\"$COMPONENT\"|"      mmms_tasks.sh
#    sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#    ./mmms_tasks.sh
#  done
#done
#
##17. GW190814 (secondary) with mu_chi1, mu_chi2 < 0.2
##    - Task J
##    - Task K
#EVENT_SAMPLES="GW190814_C01:IMRPhenomXPHM"
#POP_LABEL="pdbNG_betaSplit_brokenG"
#POP_LABEL_SUFFIX="_J"
#COMPONENT="2"
#sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
#sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
#sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#./mmms_tasks.sh &
#wait
#
#EVENT_SAMPLES="GW190814_C01:IMRPhenomXPHM"
#POP_LABEL="multiPDB_betaSplit_brokenG"
#POP_LABEL_SUFFIX="_K"
#COMPONENT="2"
#sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
#sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
#sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#./mmms_tasks.sh &
#wait
#
#18. GW230529 (primary) with mu_chi1, mu_chi2 < 0.2
#    - Task J
#    - Task K
#EVENT_SAMPLES="GW230529_Combined_PHM_highSpin"
#POP_LABEL="pdbNG_betaSplit_brokenG"
#POP_LABEL_SUFFIX="_J"
#COMPONENT="1"
#sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
#sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
#sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#./mmms_tasks.sh &
#wait
#
#EVENT_SAMPLES="GW230529_Combined_PHM_highSpin"
#POP_LABEL="multiPDB_betaSplit_brokenG"
#POP_LABEL_SUFFIX="_K"
#COMPONENT="1"
#sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
#sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
#sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#./mmms_tasks.sh &
#wait
#
#EVENT_SAMPLES="GW230529_Combined_PHM_lowSecondarySpin"
#POP_LABEL="pdbNG_betaSplit_brokenG"
#POP_LABEL_SUFFIX="_J"
#COMPONENT="1"
#sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
#sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
#sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#./mmms_tasks.sh &
#wait
#
#EVENT_SAMPLES="GW230529_Combined_PHM_lowSecondarySpin"
#POP_LABEL="multiPDB_betaSplit_brokenG"
#POP_LABEL_SUFFIX="_K"
#COMPONENT="1"
#sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
#sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
#sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#./mmms_tasks.sh &
#wait


##19. GW190814 (secondary) with mu_chi1, mu_chi2 > 0.2
##    - Task L
##    - Task M
#EVENT_SAMPLES="GW190814_C01:IMRPhenomXPHM"
#POP_LABEL="pdbNG_betaSplit_brokenG"
#POP_LABEL_SUFFIX="_L"
#COMPONENT="2"
#sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
#sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
#sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#./mmms_tasks.sh &
#wait
#
#EVENT_SAMPLES="GW190814_C01:IMRPhenomXPHM"
#POP_LABEL="multiPDB_betaSplit_brokenG"
#POP_LABEL_SUFFIX="_M"
#COMPONENT="2"
#sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
#sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
#sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#./mmms_tasks.sh &
#wait
#
#20. GW230529 (primary) with mu_chi1, mu_chi2 > 0.2
#    - Task L
#    - Task M
EVENT_SAMPLES="GW230529_Combined_PHM_highSpin"
POP_LABEL="pdbNG_betaSplit_brokenG"
POP_LABEL_SUFFIX="_L"
COMPONENT="1"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

EVENT_SAMPLES="GW230529_Combined_PHM_highSpin"
POP_LABEL="multiPDB_betaSplit_brokenG"
POP_LABEL_SUFFIX="_M"
COMPONENT="1"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

EVENT_SAMPLES="GW230529_Combined_PHM_lowSecondarySpin"
POP_LABEL="pdbNG_betaSplit_brokenG"
POP_LABEL_SUFFIX="_L"
COMPONENT="1"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

EVENT_SAMPLES="GW230529_Combined_PHM_lowSecondarySpin"
POP_LABEL="multiPDB_betaSplit_brokenG"
POP_LABEL_SUFFIX="_M"
COMPONENT="1"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
sed -i '' "s/^COMPONENT=.*/COMPONENT=\"$COMPONENT\"/" mmms_tasks.sh
sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait


##21. All events comparing default multiPDB with multiPDB singleG
#POP_LABEL_SUFFIX=""
#POP_LABEL="multiPDB_betaSplit_singleG"
#for ENTRY in "${ALL_EVENTS[@]}"; do
#  # Split into key and value parts
#  EVENT_SAMPLES=${ENTRY%%|*}
#  VALUE=${ENTRY#*|}
#  for COMPONENT in $VALUE; do
#    sed -i '' "s|^EVENT_SAMPLES=.*|EVENT_SAMPLES=\"$EVENT_SAMPLES\"|" mmms_shared_config.sh
#    sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#    sed -i '' "s|^COMPONENT=.*|COMPONENT=\"$COMPONENT\"|"      mmms_tasks.sh
#    sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#    ./mmms_tasks.sh
#  done
#done
#
##22. All events comparing default pdbNG with pdbNG singleG
#POP_LABEL_SUFFIX=""
#POP_LABEL="pdbNG_betaSplit_singleG"
#for ENTRY in "${ALL_EVENTS[@]}"; do
#  # Split into key and value parts
#  EVENT_SAMPLES=${ENTRY%%|*}
#  VALUE=${ENTRY#*|}
#  for COMPONENT in $VALUE; do
#    sed -i '' "s|^EVENT_SAMPLES=.*|EVENT_SAMPLES=\"$EVENT_SAMPLES\"|" mmms_shared_config.sh
#    sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#    sed -i '' "s|^COMPONENT=.*|COMPONENT=\"$COMPONENT\"|"      mmms_tasks.sh
#    sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#    ./mmms_tasks.sh
#  done
#done
#
##23. All events with default multiPDB but with m_spin_break = m_min
##    - Task N
#POP_LABEL_SUFFIX="_N"
#POP_LABEL="multiPDB_betaSplit_brokenG"
#for ENTRY in "${ALL_EVENTS[@]}"; do
#  # Split into key and value parts
#  EVENT_SAMPLES=${ENTRY%%|*}
#  VALUE=${ENTRY#*|}
#  for COMPONENT in $VALUE; do
#    sed -i '' "s|^EVENT_SAMPLES=.*|EVENT_SAMPLES=\"$EVENT_SAMPLES\"|" mmms_shared_config.sh
#    sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#    sed -i '' "s|^COMPONENT=.*|COMPONENT=\"$COMPONENT\"|"      mmms_tasks.sh
#    sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#    ./mmms_tasks.sh
#  done
#done
#
##24. All events with default pdnNG but with m_spin_break = m_min
##    - Task O
#POP_LABEL_SUFFIX="_O"
#POP_LABEL="pdbNG_betaSplit_brokenG"
#for ENTRY in "${ALL_EVENTS[@]}"; do
#  # Split into key and value parts
#  EVENT_SAMPLES=${ENTRY%%|*}
#  VALUE=${ENTRY#*|}
#  for COMPONENT in $VALUE; do
#    sed -i '' "s|^EVENT_SAMPLES=.*|EVENT_SAMPLES=\"$EVENT_SAMPLES\"|" mmms_shared_config.sh
#    sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#    sed -i '' "s|^COMPONENT=.*|COMPONENT=\"$COMPONENT\"|"      mmms_tasks.sh
#    sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#    ./mmms_tasks.sh
#  done
#done
#
##25. All events with default multiPDB but with m_spin_break = m_max
##    - Task P
#POP_LABEL_SUFFIX="_P"
#POP_LABEL="multiPDB_betaSplit_brokenG"
#for ENTRY in "${ALL_EVENTS[@]}"; do
#  # Split into key and value parts
#  EVENT_SAMPLES=${ENTRY%%|*}
#  VALUE=${ENTRY#*|}
#  for COMPONENT in $VALUE; do
#    sed -i '' "s|^EVENT_SAMPLES=.*|EVENT_SAMPLES=\"$EVENT_SAMPLES\"|" mmms_shared_config.sh
#    sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#    sed -i '' "s|^COMPONENT=.*|COMPONENT=\"$COMPONENT\"|"      mmms_tasks.sh
#    sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#    ./mmms_tasks.sh
#  done
#done
#
##26. All events with default pdnNG but with m_spin_break = m_max
##    - Task Q
#POP_LABEL_SUFFIX="_Q"
#POP_LABEL="pdbNG_betaSplit_brokenG"
#for ENTRY in "${ALL_EVENTS[@]}"; do
#  # Split into key and value parts
#  EVENT_SAMPLES=${ENTRY%%|*}
#  VALUE=${ENTRY#*|}
#  for COMPONENT in $VALUE; do
#    sed -i '' "s|^EVENT_SAMPLES=.*|EVENT_SAMPLES=\"$EVENT_SAMPLES\"|" mmms_shared_config.sh
#    sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#    sed -i '' "s|^COMPONENT=.*|COMPONENT=\"$COMPONENT\"|"      mmms_tasks.sh
#    sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#    ./mmms_tasks.sh
#  done
#done
##27. All events comparing default multiPDB with multiPDB betaSplit3
#POP_LABEL_SUFFIX=""
#POP_LABEL="multiPDB_betaSplit3_brokenG"
#for ENTRY in "${ALL_EVENTS[@]}"; do
#  # Split into key and value parts
#  EVENT_SAMPLES=${ENTRY%%|*}
#  VALUE=${ENTRY#*|}
#  for COMPONENT in $VALUE; do
#    sed -i '' "s|^EVENT_SAMPLES=.*|EVENT_SAMPLES=\"$EVENT_SAMPLES\"|" mmms_shared_config.sh
#    sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#    sed -i '' "s|^COMPONENT=.*|COMPONENT=\"$COMPONENT\"|"      mmms_tasks.sh
#    sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#    ./mmms_tasks.sh
#  done
#done
#
##28. All events comparing default pdbNG with pdbNG betaSplit3
#POP_LABEL_SUFFIX=""
#POP_LABEL="pdbNG_betaSplit3_brokenG"
#for ENTRY in "${ALL_EVENTS[@]}"; do
#  # Split into key and value parts
#  EVENT_SAMPLES=${ENTRY%%|*}
#  VALUE=${ENTRY#*|}
#  for COMPONENT in $VALUE; do
#    sed -i '' "s|^EVENT_SAMPLES=.*|EVENT_SAMPLES=\"$EVENT_SAMPLES\"|" mmms_shared_config.sh
#    sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#    sed -i '' "s|^COMPONENT=.*|COMPONENT=\"$COMPONENT\"|"      mmms_tasks.sh
#    sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#    ./mmms_tasks.sh
#  done
#done
#
##29. All events comparing default pdbNG with sig_peak1_large
#POP_LABEL_SUFFIX=""
#POP_LABEL="pdbNG_betaSplit_brokenG_sig_peak1_large"
#for ENTRY in "${ALL_EVENTS[@]}"; do
#  # Split into key and value parts
#  EVENT_SAMPLES=${ENTRY%%|*}
#  VALUE=${ENTRY#*|}
#  for COMPONENT in $VALUE; do
#    sed -i '' "s|^EVENT_SAMPLES=.*|EVENT_SAMPLES=\"$EVENT_SAMPLES\"|" mmms_shared_config.sh
#    sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#    sed -i '' "s|^COMPONENT=.*|COMPONENT=\"$COMPONENT\"|"      mmms_tasks.sh
#    sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#    ./mmms_tasks.sh
#  done
#done
#
##30. All events comparing default pdbNG with sig_peak1_test
#POP_LABEL_SUFFIX=""
#POP_LABEL="pdbNG_betaSplit_brokenG_sig_peak1_test"
#for ENTRY in "${ALL_EVENTS[@]}"; do
#  # Split into key and value parts
#  EVENT_SAMPLES=${ENTRY%%|*}
#  VALUE=${ENTRY#*|}
#  for COMPONENT in $VALUE; do
#    sed -i '' "s|^EVENT_SAMPLES=.*|EVENT_SAMPLES=\"$EVENT_SAMPLES\"|" mmms_shared_config.sh
#    sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#    sed -i '' "s|^COMPONENT=.*|COMPONENT=\"$COMPONENT\"|"      mmms_tasks.sh
#    sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#    ./mmms_tasks.sh
#  done
#done
#
##31. All events comparing default pdbNG with _tight_prior
#POP_LABEL_SUFFIX=""
#POP_LABEL="pdbNG_betaSplit_brokenG_tight_prior"
#for ENTRY in "${ALL_EVENTS[@]}"; do
#  # Split into key and value parts
#  EVENT_SAMPLES=${ENTRY%%|*}
#  VALUE=${ENTRY#*|}
#  for COMPONENT in $VALUE; do
#    sed -i '' "s|^EVENT_SAMPLES=.*|EVENT_SAMPLES=\"$EVENT_SAMPLES\"|" mmms_shared_config.sh
#    sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#    sed -i '' "s|^COMPONENT=.*|COMPONENT=\"$COMPONENT\"|"      mmms_tasks.sh
#    sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#    ./mmms_tasks.sh
#  done
#done
#
##32. All events comparing default pdbNG with _same_mbrk
#POP_LABEL_SUFFIX=""
#POP_LABEL="pdbNG_betaSplit_brokenG_same_mbrk"
#for ENTRY in "${ALL_EVENTS[@]}"; do
#  # Split into key and value parts
#  EVENT_SAMPLES=${ENTRY%%|*}
#  VALUE=${ENTRY#*|}
#  for COMPONENT in $VALUE; do
#    sed -i '' "s|^EVENT_SAMPLES=.*|EVENT_SAMPLES=\"$EVENT_SAMPLES\"|" mmms_shared_config.sh
#    sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#    sed -i '' "s|^COMPONENT=.*|COMPONENT=\"$COMPONENT\"|"      mmms_tasks.sh
#    sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#    ./mmms_tasks.sh
#  done
#done
#
##33. All events comparing default pdbNG (mu_costilt = 0) with pdbNG with mu_costilt = 0 and mu_costilt = -1
##    - Task U
##    - Task V
#POP_LABEL_SUFFIX="_U"
#POP_LABEL="pdbNG_betaSplit_brokenG"
#for ENTRY in "${ALL_EVENTS[@]}"; do
#  # Split into key and value parts
#  EVENT_SAMPLES=${ENTRY%%|*}
#  VALUE=${ENTRY#*|}
#  for COMPONENT in $VALUE; do
#    sed -i '' "s|^EVENT_SAMPLES=.*|EVENT_SAMPLES=\"$EVENT_SAMPLES\"|" mmms_shared_config.sh
#    sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#    sed -i '' "s|^COMPONENT=.*|COMPONENT=\"$COMPONENT\"|"      mmms_tasks.sh
#    sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#    ./mmms_tasks.sh
#  done
#done
#
#POP_LABEL_SUFFIX="_V"
#POP_LABEL="pdbNG_betaSplit_brokenG"
#for ENTRY in "${ALL_EVENTS[@]}"; do
#  # Split into key and value parts
#  EVENT_SAMPLES=${ENTRY%%|*}
#  VALUE=${ENTRY#*|}
#  for COMPONENT in $VALUE; do
#    sed -i '' "s|^EVENT_SAMPLES=.*|EVENT_SAMPLES=\"$EVENT_SAMPLES\"|" mmms_shared_config.sh
#    sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
#    sed -i '' "s|^COMPONENT=.*|COMPONENT=\"$COMPONENT\"|"      mmms_tasks.sh
#    sed -i '' "s/^POP_LABEL=.*/POP_LABEL=\"$POP_LABEL\"/" mmms_tasks.sh
#    ./mmms_tasks.sh
#  done
#done
#
#wait
#exit 0