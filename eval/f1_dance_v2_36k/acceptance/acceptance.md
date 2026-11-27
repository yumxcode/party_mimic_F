# Sim2sim acceptance — f1_dance_v2_36k (MuJoCo CPU, A1–A5 of docs/acceptance_plan.md)

Stance run length for A4: 8 frames @ 50 Hz (= 5 frames @ 30 fps). Overall: **FAIL**

| clip | kind | frames | A1 fall | A2 local err mm (limit) | A3 joint err rad (limit) | A4 flat % L/R (limit) | A5 adduct % (limit) | pass |
|---|---|---|---|---|---|---|---|---|
| dance1_subject3_255_706 | trained | 750/750 | no | 11.9 (50) | 0.0376 (0.2) | 95.7/81.3 (90) | 0.00 (3.0) | ✗ |
| dance2_subject1_352_795 | trained | 737/737 | no | 9.4 (50) | 0.0356 (0.2) | 92.3/94.3 (90) | 0.27 (3.0) | ✓ |
| dance2_subject4_2835_2925 | trained | 149/149 | no | 9.6 (50) | 0.0356 (0.2) | 100.0/100.0 (90) | 0.00 (3.0) | ✓ |
| dance2_subject3_300_450 | held-out | 249/249 | no | 44.4 (80) | 0.0709 (0.2) | 95.6/98.8 (90) | 0.00 (3.0) | ✓ |

A4 stance frames per clip (gate 7 sample size, minimum 10): dance1_subject3_255_706: 750/750; dance2_subject1_352_795: 731/732; dance2_subject4_2835_2925: 149/149; dance2_subject3_300_450: 249/249

A4 with the checker's default stance run (5 frames @ 50 Hz, sensitivity only): dance1_subject3_255_706: 95.7/81.3; dance2_subject1_352_795: 92.3/94.3; dance2_subject4_2835_2925: 100.0/100.0; dance2_subject3_300_450: 95.6/98.8
