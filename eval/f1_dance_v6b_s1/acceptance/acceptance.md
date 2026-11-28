# Sim2sim acceptance — f1_dance_v6b_s1 (MuJoCo CPU, A1–A5 of docs/acceptance_plan.md)

Stance run length for A4: 8 frames @ 50 Hz (= 5 frames @ 30 fps). Overall: **FAIL**

| clip | kind | frames | A1 fall | A2 local err mm (limit) | A3 joint err rad (limit) | A4 flat % L/R (limit) | A5 adduct % (limit) | pass |
|---|---|---|---|---|---|---|---|---|
| dance1_subject3_255_706 | trained | 750/750 | no | 9.3 (50) | 0.0326 (0.2) | 97.7/96.9 (90) | 0.00 (3.0) | ✓ |
| dance2_subject1_352_795 | trained | 737/737 | no | 9.4 (50) | 0.0362 (0.2) | 97.2/98.8 (90) | 0.68 (3.0) | ✓ |
| dance2_subject4_2835_2925 | trained | 146/149 | FELL@145 | 16.0 (50) | 0.0467 (0.2) | 99.3/96.6 (90) | 0.00 (3.0) | ✗ |
| dance2_subject3_300_450 | held-out | 249/249 | no | 39.1 (80) | 0.0574 (0.2) | 94.4/100.0 (90) | 0.00 (3.0) | ✓ |

A4 stance frames per clip (gate 7 sample size, minimum 10): dance1_subject3_255_706: 750/747; dance2_subject1_352_795: 712/725; dance2_subject4_2835_2925: 141/145; dance2_subject3_300_450: 249/249

A4 with the checker's default stance run (5 frames @ 50 Hz, sensitivity only): dance1_subject3_255_706: 97.7/96.9; dance2_subject1_352_795: 97.2/98.8; dance2_subject4_2835_2925: 99.3/96.6; dance2_subject3_300_450: 94.4/100.0
