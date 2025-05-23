#!/bin/bash

ALL_EVENTS=(
"GW230529_Combined_PHM_highSpin"
"GW230529_Combined_PHM_lowSecondarySpin"
"GW190425_C01:IMRPhenomPv2_NRTidal:HighSpin"
"GW190814_C01:IMRPhenomXPHM"
"GW190917_C01:IMRPhenomXPHM"
"GW200105_C01:IMRPhenomXPHM"
"GW200115_C01:IMRPhenomNSBH:HighSpin"
)

#Example loop over all events
#
#for EVENT_SAMPLES in "${EVENT_SAMPLES_LIST[@]}"; do
#  sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
#
#  echo "Running scripts for EVENT_SAMPLES=$EVENT_SAMPLES"


#1. GW230529 (primary and secondary) with varying gamma_low, eta_low, and all other params fixed to MAP
#    - Task A
EVENT_SAMPLES="GW230529_Combined_PHM_highSpin"
POP_LABEL_SUFFIX="_A"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

EVENT_SAMPLES="GW230529_Combined_PHM_lowSecondarySpin"
POP_LABEL_SUFFIX="_A"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

#2. GW190814 (primary and secondary) with varying gamma_low, eta_low, and all other params fixed to MAP
#    - Task A
EVENT_SAMPLES="GW190814_C01:IMRPhenomXPHM"
POP_LABEL_SUFFIX="_A"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

#3. GW190425 (primary and secondary) with varying gamma_low, eta_low, and all other params fixed to MAP
#    - Task A
EVENT_SAMPLES="GW190425_C01:IMRPhenomPv2_NRTidal:HighSpin"
POP_LABEL_SUFFIX="_A"
sed -i '' "s/^EVENT_SAMPLES=.*/EVENT_SAMPLES=\"$EVENT_SAMPLES\"/" mmms_shared_config.sh
sed -i '' "s/^POP_LABEL_SUFFIX=.*/POP_LABEL_SUFFIX=\"$POP_LABEL_SUFFIX\"/" mmms_tasks.sh
./mmms_tasks.sh &
wait

#4. GW230529 (primary and secondary) with fixed gamma_high, eta_high
#    - Task B
#    - Task T


#5. GW200115 (primary and secondary) with fixed gamma_high, eta_high
#    - Task B
#    - Task T


#6. All events (primary and secondary) compare default multiPDB vs default pdbNG


#7. All events (primary and secondary) with default pdbNG but only sig_peak_NS < median
#    - Task C


#8. GW230529 (primary and secondary) with fixed mu_peak_NS to Galactic BNS rate
#    - Task D and E


#9. GW190814 (primary and secondary) with fixed mu_peak_NS to Galactic BNS rate
#    - Task D and E


#10. GW200115 (primary and secondary) with fixed mu_peak_NS to Galactic BNS rate
#    - Task D and E


#11. GW190814 (secondary) with fixed beta_1 = beta_2 = 0
#    - Task F
#    - Task G
#    - Task R
#    - Task S


#12. GW230529 (primary) with fixed beta_1 = beta_2 = 0
#    - Task F
#    - Task G
#    - Task R
#    - Task S


#13. GW190814 (secondary) with fixed beta_1 = beta_2 = 4
#    - Task H
#    - Task I


#14. GW230529 (primary) with fixed beta_1 = beta_2 = 4
#    - Task H
#    - Task I


#15. All events (primary and secondary) with smooth pairing function for multiPDB


#16. All events (primary and secondary) with smooth pairing function for pdbNG


#17. GW190814 (primary and secondary) with mu_chi1, mu_chi2 < 0.2
#    - Task J
#    - Task K


#18. GW230529 (primary and secondary) with mu_chi1, mu_chi2 < 0.2
#    - Task J
#    - Task K


#19. GW190814 (primary and secondary) with mu_chi1, mu_chi2 > 0.2
#    - Task L
#    - Task M


#20. GW230529 (primary and secondary) with mu_chi1, mu_chi2 > 0.2
#    - Task L
#    - Task M


#21. All events (primary and secondary) comparing default multiPDB with multiPDB singleG


#22. All events (primary and secondary) comparing default pdbNG with pdbNG singleG


#23. All events (primary and secondary) with default multiPDB but with m_spin_break = m_min
#    - Task N


#24. All events (primary and secondary) with default pdnNG but with m_spin_break = m_min
#    - Task O


#23. All events (primary and secondary) with default multiPDB but with m_spin_break = m_max
#    - Task P


#24. All events (primary and secondary) with default pdnNG but with m_spin_break = m_max
#    - Task Q
exit 0