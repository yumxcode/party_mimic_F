# Sim2sim acceptance — f1_dance_v1 (MuJoCo CPU, A1–A5 of docs/acceptance_plan.md)

Stance run length for A4: 8 frames @ 50 Hz (= 5 frames @ 30 fps). Overall: **FAIL**

| clip | kind | frames | A1 fall | A2 local err mm (limit) | A3 joint err rad (limit) | A4 flat % L/R (limit) | A5 adduct % (limit) | pass |
|---|---|---|---|---|---|---|---|---|
| dance1_subject3_255_706 | trained | 32/750 | FELL@31 | 149.6 (50) | 0.1738 (0.2) | 33.3/71.9 (90) | 0.00 (3.0) | ✗ |
| dance2_subject1_352_795 | trained | 19/737 | FELL@18 | 93.8 (50) | 0.1037 (0.2) | 63.6/50.0 (90) | 0.00 (3.0) | ✗ |
| dance2_subject4_2835_2925 | trained | 17/149 | FELL@16 | 109.1 (50) | 0.1482 (0.2) | 33.3/47.1 (90) | 0.00 (3.0) | ✗ |
| dance2_subject3_300_450 | held-out | 16/249 | FELL@15 | 83.7 (80) | 0.0956 (0.2) | 85.7/25.0 (90) | 0.00 (3.0) | ✗ |

A4 stance frames per clip (gate 7 sample size, minimum 10): dance1_subject3_255_706: 18/32; dance2_subject1_352_795: 11/10; dance2_subject4_2835_2925: 15/17; dance2_subject3_300_450: 14/12

A4 with the checker's default stance run (5 frames @ 50 Hz, sensitivity only): dance1_subject3_255_706: 33.3/71.9; dance2_subject1_352_795: 47.1/52.9; dance2_subject4_2835_2925: 33.3/47.1; dance2_subject3_300_450: 85.7/25.0
