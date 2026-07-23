# Tempering accuracy — 2026-07-22

**Verdict:** `current_rwm_not_eligible`

[Gates](2026-07-22-tempering-accuracy-gates.png) · [Cost](2026-07-22-tempering-accuracy-cost.png)

## Evidence

| Item | Value |
| --- | --- |
| Execution | 508 / 508 |
| Source | `4d55dd5f9d1f`; clean |
| Host | Apple M3 Pro; Darwin 26.2 arm64 |
| Python | CPython 3.13.9 |
| Raw leaves | 508 |

## Gates

| Family | Registered | Evaluated | Passed |
| --- | --- | --- | --- |
| Centering | 6,228 | 6,228 | 5,001 |
| Evidence resolution | 84 | 84 | 28 |

## Minimum passing sweep

| Geometry | d | N | Lane | Sweeps |
| --- | --- | --- | --- | --- |
| G0 | 4 | 1,000 | cpu_f64 | 5 |
| G0 | 4 | 1,000 | mps_f32 | 5 |
| G0 | 4 | 10,000 | cpu_f64 | 5 |
| G0 | 4 | 10,000 | mps_f32 | 5 |
| G0 | 32 | 1,000 | cpu_f64 | — |
| G0 | 32 | 1,000 | mps_f32 | — |
| G0 | 32 | 10,000 | cpu_f64 | 20 |
| G0 | 32 | 10,000 | mps_f32 | 50 |
| G0 | 128 | 1,000 | cpu_f64 | — |
| G0 | 128 | 1,000 | mps_f32 | — |
| G0 | 128 | 10,000 | cpu_f64 | — |
| G0 | 128 | 10,000 | mps_f32 | — |
| G1 | 4 | 1,000 | cpu_f64 | 20 |
| G1 | 4 | 1,000 | mps_f32 | 5 |
| G1 | 4 | 10,000 | cpu_f64 | 5 |
| G1 | 4 | 10,000 | mps_f32 | 5 |
| G1 | 32 | 1,000 | cpu_f64 | — |
| G1 | 32 | 1,000 | mps_f32 | — |
| G1 | 32 | 10,000 | cpu_f64 | 50 |
| G1 | 32 | 10,000 | mps_f32 | — |
| G1 | 128 | 1,000 | cpu_f64 | — |
| G1 | 128 | 1,000 | mps_f32 | — |
| G1 | 128 | 10,000 | cpu_f64 | — |
| G1 | 128 | 10,000 | mps_f32 | — |

## Matched challenge

| Geometry | d | N | Lane | Systematic | Multinomial | Mean RMSE S / M | Cov RMSE S / M | Evidence RMSE S / M |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G0 | 32 | 1,000 | cpu_f64 | indeterminate_evidence | indeterminate_evidence | 0.0512331 / 0.0584545 | 0.196653 / 0.210311 | 0.563848 / 0.598614 |
| G0 | 32 | 1,000 | mps_f32 | indeterminate_evidence | indeterminate_evidence | 0.0514053 / 0.0589801 | 0.195622 / 0.208082 | 0.579716 / 0.728917 |
| G0 | 128 | 1,000 | cpu_f64 | indeterminate_evidence | indeterminate_evidence | 0.321366 / 0.32908 | 0.685806 / 0.70639 | 4.8031e+09 / 6.7463e+09 |
| G0 | 128 | 1,000 | mps_f32 | indeterminate_evidence | indeterminate_evidence | 0.317155 / 0.332484 | 0.684834 / 0.706992 | 7.29239e+09 / 2.95458e+10 |
| G0 | 128 | 10,000 | cpu_f64 | indeterminate_evidence | indeterminate_evidence | 0.170833 / 0.169109 | 0.425517 / 0.441087 | 603118 / 754830 |
| G0 | 128 | 10,000 | mps_f32 | indeterminate_evidence | indeterminate_evidence | 0.167304 / 0.171501 | 0.426876 / 0.444497 | 578627 / 319115 |
| G1 | 32 | 1,000 | cpu_f64 | indeterminate_evidence | indeterminate_evidence | 0.0496206 / 0.0576948 | 0.127118 / 0.135729 | 3.33557 / 4.73695 |
| G1 | 32 | 1,000 | mps_f32 | indeterminate_evidence | indeterminate_evidence | 0.051621 / 0.0599639 | 0.129382 / 0.137133 | 3.04583 / 4.35482 |
| G1 | 128 | 1,000 | cpu_f64 | indeterminate_evidence | indeterminate_evidence | 0.233034 / 0.251559 | 0.834795 / 0.856666 | 1.2046e+32 / 5.2237e+32 |
| G1 | 128 | 1,000 | mps_f32 | indeterminate_evidence | indeterminate_evidence | 0.239912 / 0.254126 | 0.837408 / 0.857111 | 4.28467e+31 / 1.33162e+33 |
| G1 | 128 | 10,000 | cpu_f64 | indeterminate_evidence | indeterminate_evidence | 0.157817 / 0.159833 | 0.513571 / 0.525496 | 2.29798e+18 / 2.59451e+18 |
| G1 | 128 | 10,000 | mps_f32 | indeterminate_evidence | indeterminate_evidence | 0.152773 / 0.15429 | 0.511764 / 0.532591 | 4.71967e+18 / 1.19449e+19 |

## All cells

| Cell | Status | Mean RMSE | Cov RMSE | Evidence RMSE | Stages | Pairs | First s | Steady s | RSS MiB | MPS MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_systematic-g0-d4-n1000-cpu_f64-systematic-s5 | eligible | 0.0331619 | 0.0731691 | 0.0478375 | 2 | 11,000 | 1.08737 | 0.224177 | 389 | — |
| current_systematic-g0-d4-n1000-mps_f32-systematic-s5 | eligible | 0.036334 | 0.0802321 | 0.0513586 | 2 | 11,000 | 0.395814 | 0.127524 | 303 | 0.768948 |
| current_systematic-g0-d4-n1000-cpu_f64-systematic-s20 | eligible | 0.0294003 | 0.0697425 | 0.0378885 | 2 | 41,000 | 1.90614 | 1.02378 | 625.641 | — |
| current_systematic-g0-d4-n1000-mps_f32-systematic-s20 | eligible | 0.0310486 | 0.0723234 | 0.0372318 | 2 | 41,000 | 0.607712 | 0.191454 | 312.078 | 1.05519 |
| current_systematic-g0-d4-n1000-cpu_f64-systematic-s50 | eligible | 0.0274253 | 0.0711853 | 0.0460298 | 2 | 101,000 | 5.4734 | 4.50492 | 1698.77 | — |
| current_systematic-g0-d4-n1000-mps_f32-systematic-s50 | eligible | 0.0332143 | 0.072597 | 0.0395178 | 2 | 101,000 | 0.776371 | 0.329108 | 328.766 | 1.18567 |
| current_systematic-g0-d4-n10000-cpu_f64-systematic-s5 | eligible | 0.0111507 | 0.02235 | 0.0141033 | 2 | 110,000 | 1.10963 | 0.238731 | 392.172 | — |
| current_systematic-g0-d4-n10000-mps_f32-systematic-s5 | eligible | 0.0113677 | 0.0230433 | 0.0175524 | 2 | 110,000 | 0.403585 | 0.130106 | 306.625 | 10.7388 |
| current_systematic-g0-d4-n10000-cpu_f64-systematic-s20 | eligible | 0.00979149 | 0.0228982 | 0.0122075 | 2 | 410,000 | 1.64976 | 0.768453 | 589.25 | — |
| current_systematic-g0-d4-n10000-mps_f32-systematic-s20 | eligible | 0.00999732 | 0.0229076 | 0.0131247 | 2 | 410,000 | 0.464249 | 0.196047 | 316.297 | 13.3857 |
| current_systematic-g0-d4-n10000-cpu_f64-systematic-s50 | eligible | 0.0101498 | 0.0226446 | 0.00977483 | 2 | 1,010,000 | 3.6519 | 2.75768 | 985.828 | — |
| current_systematic-g0-d4-n10000-mps_f32-systematic-s50 | eligible | 0.00983143 | 0.0206259 | 0.0133257 | 2 | 1,010,000 | 0.57874 | 0.330965 | 335.094 | 13.8871 |
| current_systematic-g0-d32-n1000-cpu_f64-systematic-s5 | indeterminate_evidence | 0.222306 | 0.444198 | 5.33591 | 6 | 31,000 | — | — | — | — |
| current_systematic-g0-d32-n1000-mps_f32-systematic-s5 | indeterminate_evidence | 0.219296 | 0.44604 | 2.4132 | 6 | 31,000 | — | — | — | — |
| current_systematic-g0-d32-n1000-cpu_f64-systematic-s20 | indeterminate_evidence | 0.0512331 | 0.196653 | 0.563848 | 6 | 121,000 | — | — | — | — |
| current_systematic-g0-d32-n1000-mps_f32-systematic-s20 | indeterminate_evidence | 0.0514053 | 0.195622 | 0.579716 | 6 | 121,000 | — | — | — | — |
| current_systematic-g0-d32-n1000-cpu_f64-systematic-s50 | indeterminate_evidence | 0.0332598 | 0.180292 | 0.227968 | 6 | 301,000 | — | — | — | — |
| current_systematic-g0-d32-n1000-mps_f32-systematic-s50 | indeterminate_evidence | 0.0329256 | 0.180417 | 0.273425 | 6 | 301,000 | — | — | — | — |
| current_systematic-g0-d32-n10000-cpu_f64-systematic-s5 | indeterminate_evidence | 0.136999 | 0.291002 | 0.652145 | 6 | 310,000 | — | — | — | — |
| current_systematic-g0-d32-n10000-mps_f32-systematic-s5 | indeterminate_evidence | 0.144867 | 0.295526 | 1.5114 | 6 | 310,000 | — | — | — | — |
| current_systematic-g0-d32-n10000-cpu_f64-systematic-s20 | eligible | 0.0167292 | 0.0628291 | 0.0889813 | 6 | 1,210,000 | 1.91854 | 1.01069 | 571.969 | — |
| current_systematic-g0-d32-n10000-mps_f32-systematic-s20 | indeterminate_evidence | 0.0174283 | 0.0625601 | 0.121197 | 6 | 1,210,000 | — | — | — | — |
| current_systematic-g0-d32-n10000-cpu_f64-systematic-s50 | eligible | 0.010023 | 0.0574564 | 0.0430341 | 6 | 3,010,000 | 4.20121 | 3.2624 | 1011.12 | — |
| current_systematic-g0-d32-n10000-mps_f32-systematic-s50 | eligible | 0.0101492 | 0.0575952 | 0.0337956 | 6 | 3,010,000 | 0.997291 | 0.740305 | 393.344 | 208.569 |
| current_systematic-g0-d128-n1000-cpu_f64-systematic-s5 | indeterminate_evidence | 0.53934 | 0.911038 | 5.92492e+10 | 10 | 51,000 | — | — | — | — |
| current_systematic-g0-d128-n1000-mps_f32-systematic-s5 | indeterminate_evidence | 0.534113 | 0.910799 | 5.96469e+11 | 10 | 51,000 | — | — | — | — |
| current_systematic-g0-d128-n1000-cpu_f64-systematic-s20 | indeterminate_evidence | 0.321366 | 0.685806 | 4.8031e+09 | 11 | 221,000 | — | — | — | — |
| current_systematic-g0-d128-n1000-mps_f32-systematic-s20 | indeterminate_evidence | 0.317155 | 0.684834 | 7.29239e+09 | 11 | 221,000 | — | — | — | — |
| current_systematic-g0-d128-n1000-cpu_f64-systematic-s50 | indeterminate_evidence | 0.121533 | 0.462643 | 3.67689e+06 | 12 | 601,000 | — | — | — | — |
| current_systematic-g0-d128-n1000-mps_f32-systematic-s50 | indeterminate_evidence | 0.120205 | 0.464257 | 5.67168e+06 | 12 | 601,000 | — | — | — | — |
| current_systematic-g0-d128-n10000-cpu_f64-systematic-s5 | indeterminate_evidence | 0.415253 | 0.829404 | 7.11256e+08 | 11 | 560,000 | — | — | — | — |
| current_systematic-g0-d128-n10000-mps_f32-systematic-s5 | indeterminate_evidence | 0.417522 | 0.830399 | 1.54584e+09 | 11 | 560,000 | — | — | — | — |
| current_systematic-g0-d128-n10000-cpu_f64-systematic-s20 | indeterminate_evidence | 0.170833 | 0.425517 | 603118 | 12 | 2,410,000 | — | — | — | — |
| current_systematic-g0-d128-n10000-mps_f32-systematic-s20 | indeterminate_evidence | 0.167304 | 0.426876 | 578627 | 12 | 2,410,000 | — | — | — | — |
| current_systematic-g0-d128-n10000-cpu_f64-systematic-s50 | indeterminate_evidence | 0.0382676 | 0.164753 | 118.609 | 13 | 6,510,000 | — | — | — | — |
| current_systematic-g0-d128-n10000-mps_f32-systematic-s50 | indeterminate_evidence | 0.0385125 | 0.16391 | 121.358 | 13 | 6,510,000 | — | — | — | — |
| current_systematic-g1-d4-n1000-cpu_f64-systematic-s5 | indeterminate_evidence | 0.0397588 | 0.0491467 | 0.100692 | 4 | 21,000 | — | — | — | — |
| current_systematic-g1-d4-n1000-mps_f32-systematic-s5 | eligible | 0.0393047 | 0.0546604 | 0.0806629 | 4 | 21,000 | 0.619982 | 0.304848 | 328.578 | 0.907478 |
| current_systematic-g1-d4-n1000-cpu_f64-systematic-s20 | eligible | 0.0331492 | 0.0478761 | 0.0709958 | 4 | 81,000 | 8.81539 | 7.80956 | 982.391 | — |
| current_systematic-g1-d4-n1000-mps_f32-systematic-s20 | eligible | 0.0297794 | 0.0555686 | 0.0544765 | 4 | 81,000 | 0.715356 | 0.404814 | 339.609 | 1.04033 |
| current_systematic-g1-d4-n1000-cpu_f64-systematic-s50 | eligible | 0.0306363 | 0.0459739 | 0.0531325 | 4 | 201,000 | 43.1739 | 41.7828 | 4430.38 | — |
| current_systematic-g1-d4-n1000-mps_f32-systematic-s50 | eligible | 0.0352098 | 0.0460293 | 0.0619865 | 4 | 201,000 | 1.119 | 0.622628 | 360.359 | 1.11457 |
| current_systematic-g1-d4-n10000-cpu_f64-systematic-s5 | eligible | 0.0106168 | 0.0154615 | 0.0318367 | 4 | 210,000 | 1.28876 | 0.276493 | 415.5 | — |
| current_systematic-g1-d4-n10000-mps_f32-systematic-s5 | eligible | 0.0113631 | 0.0181504 | 0.0297652 | 4 | 210,000 | 0.632614 | 0.312012 | 333.891 | 9.70744 |
| current_systematic-g1-d4-n10000-cpu_f64-systematic-s20 | eligible | 0.0097399 | 0.0180157 | 0.017932 | 4 | 810,000 | 1.8797 | 0.861574 | 621.797 | — |
| current_systematic-g1-d4-n10000-mps_f32-systematic-s20 | eligible | 0.0104685 | 0.0151082 | 0.0165563 | 4 | 810,000 | 0.724588 | 0.413132 | 344.094 | 11.358 |
| current_systematic-g1-d4-n10000-cpu_f64-systematic-s50 | eligible | 0.00881028 | 0.0153432 | 0.0190946 | 4 | 2,010,000 | 4.01546 | 3.01431 | 1021.02 | — |
| current_systematic-g1-d4-n10000-mps_f32-systematic-s50 | eligible | 0.010393 | 0.0166717 | 0.0165812 | 4 | 2,010,000 | 0.924651 | 0.630104 | 366.016 | 16.1335 |
| current_systematic-g1-d32-n1000-cpu_f64-systematic-s5 | indeterminate_evidence | 0.29607 | 0.492194 | 1235.01 | 12 | 61,000 | — | — | — | — |
| current_systematic-g1-d32-n1000-mps_f32-systematic-s5 | indeterminate_evidence | 0.293852 | 0.493957 | 512.915 | 12 | 61,000 | — | — | — | — |
| current_systematic-g1-d32-n1000-cpu_f64-systematic-s20 | indeterminate_evidence | 0.0496206 | 0.127118 | 3.33557 | 12 | 241,000 | — | — | — | — |
| current_systematic-g1-d32-n1000-mps_f32-systematic-s20 | indeterminate_evidence | 0.051621 | 0.129382 | 3.04583 | 12 | 241,000 | — | — | — | — |
| current_systematic-g1-d32-n1000-cpu_f64-systematic-s50 | indeterminate_evidence | 0.0313018 | 0.10583 | 0.950863 | 13 | 651,000 | — | — | — | — |
| current_systematic-g1-d32-n1000-mps_f32-systematic-s50 | indeterminate_evidence | 0.0328062 | 0.1067 | 0.965777 | 12 | 601,000 | — | — | — | — |
| current_systematic-g1-d32-n10000-cpu_f64-systematic-s5 | indeterminate_evidence | 0.196987 | 0.286917 | 20.2234 | 12 | 610,000 | — | — | — | — |
| current_systematic-g1-d32-n10000-mps_f32-systematic-s5 | indeterminate_evidence | 0.203116 | 0.273727 | 5.49562 | 12 | 610,000 | — | — | — | — |
| current_systematic-g1-d32-n10000-cpu_f64-systematic-s20 | indeterminate_evidence | 0.0164563 | 0.0367964 | 0.220862 | 13 | 2,610,000 | — | — | — | — |
| current_systematic-g1-d32-n10000-mps_f32-systematic-s20 | indeterminate_evidence | 0.0152881 | 0.036684 | 0.393081 | 13 | 2,610,000 | — | — | — | — |
| current_systematic-g1-d32-n10000-cpu_f64-systematic-s50 | eligible | 0.0105476 | 0.0340635 | 0.0867814 | 13 | 6,510,000 | 5.35181 | 4.24278 | 1045.75 | — |
| current_systematic-g1-d32-n10000-mps_f32-systematic-s50 | failed_accuracy | 0.00984926 | 0.0337545 | 0.0820232 | 13 | 6,510,000 | — | — | — | — |
| current_systematic-g1-d128-n1000-cpu_f64-systematic-s5 | indeterminate_evidence | 0.47403 | 0.975583 | 4.00357e+37 | 17 | 86,000 | — | — | — | — |
| current_systematic-g1-d128-n1000-mps_f32-systematic-s5 | indeterminate_evidence | 0.468069 | 0.976907 | 1.00522e+40 | 17 | 86,000 | — | — | — | — |
| current_systematic-g1-d128-n1000-cpu_f64-systematic-s20 | indeterminate_evidence | 0.233034 | 0.834795 | 1.2046e+32 | 21 | 421,000 | — | — | — | — |
| current_systematic-g1-d128-n1000-mps_f32-systematic-s20 | indeterminate_evidence | 0.239912 | 0.837408 | 4.28467e+31 | 21 | 421,000 | — | — | — | — |
| current_systematic-g1-d128-n1000-cpu_f64-systematic-s50 | indeterminate_evidence | 0.0805284 | 0.422481 | 1.58067e+20 | 25 | 1,251,000 | — | — | — | — |
| current_systematic-g1-d128-n1000-mps_f32-systematic-s50 | indeterminate_evidence | 0.0815281 | 0.425945 | 1.06262e+20 | 25 | 1,251,000 | — | — | — | — |
| current_systematic-g1-d128-n10000-cpu_f64-systematic-s5 | indeterminate_evidence | 0.368837 | 0.938795 | 1.5103e+34 | 20 | 1,010,000 | — | — | — | — |
| current_systematic-g1-d128-n10000-mps_f32-systematic-s5 | indeterminate_evidence | 0.365226 | 0.937638 | 1.13855e+34 | 20 | 1,010,000 | — | — | — | — |
| current_systematic-g1-d128-n10000-cpu_f64-systematic-s20 | indeterminate_evidence | 0.157817 | 0.513571 | 2.29798e+18 | 24 | 4,810,000 | — | — | — | — |
| current_systematic-g1-d128-n10000-mps_f32-systematic-s20 | indeterminate_evidence | 0.152773 | 0.511764 | 4.71967e+18 | 24 | 4,810,000 | — | — | — | — |
| current_systematic-g1-d128-n10000-cpu_f64-systematic-s50 | indeterminate_evidence | 0.0390377 | 0.137358 | 5.43159e+06 | 26 | 13,010,000 | — | — | — | — |
| current_systematic-g1-d128-n10000-mps_f32-systematic-s50 | indeterminate_evidence | 0.0392666 | 0.135487 | 5.48874e+06 | 27 | 13,510,000 | — | — | — | — |
| matched_multinomial-g0-d32-n1000-cpu_f64-multinomial-s20 | indeterminate_evidence | 0.0584545 | 0.210311 | 0.598614 | 6 | 121,000 | — | — | — | — |
| matched_multinomial-g0-d32-n1000-mps_f32-multinomial-s20 | indeterminate_evidence | 0.0589801 | 0.208082 | 0.728917 | 6 | 121,000 | — | — | — | — |
| matched_multinomial-g0-d128-n1000-cpu_f64-multinomial-s20 | indeterminate_evidence | 0.32908 | 0.70639 | 6.7463e+09 | 11 | 221,000 | — | — | — | — |
| matched_multinomial-g0-d128-n1000-mps_f32-multinomial-s20 | indeterminate_evidence | 0.332484 | 0.706992 | 2.95458e+10 | 11 | 221,000 | — | — | — | — |
| matched_multinomial-g0-d128-n10000-cpu_f64-multinomial-s20 | indeterminate_evidence | 0.169109 | 0.441087 | 754830 | 12 | 2,410,000 | — | — | — | — |
| matched_multinomial-g0-d128-n10000-mps_f32-multinomial-s20 | indeterminate_evidence | 0.171501 | 0.444497 | 319115 | 12 | 2,410,000 | — | — | — | — |
| matched_multinomial-g1-d32-n1000-cpu_f64-multinomial-s20 | indeterminate_evidence | 0.0576948 | 0.135729 | 4.73695 | 12 | 241,000 | — | — | — | — |
| matched_multinomial-g1-d32-n1000-mps_f32-multinomial-s20 | indeterminate_evidence | 0.0599639 | 0.137133 | 4.35482 | 12 | 241,000 | — | — | — | — |
| matched_multinomial-g1-d128-n1000-cpu_f64-multinomial-s20 | indeterminate_evidence | 0.251559 | 0.856666 | 5.2237e+32 | 20 | 401,000 | — | — | — | — |
| matched_multinomial-g1-d128-n1000-mps_f32-multinomial-s20 | indeterminate_evidence | 0.254126 | 0.857111 | 1.33162e+33 | 20 | 401,000 | — | — | — | — |
| matched_multinomial-g1-d128-n10000-cpu_f64-multinomial-s20 | indeterminate_evidence | 0.159833 | 0.525496 | 2.59451e+18 | 24 | 4,810,000 | — | — | — | — |
| matched_multinomial-g1-d128-n10000-mps_f32-multinomial-s20 | indeterminate_evidence | 0.15429 | 0.532591 | 1.19449e+19 | 24 | 4,810,000 | — | — | — | — |

## Failures

None.

## Attempts

None.

## Exclusions

| Arm | Status | Issue |
| --- | --- | --- |
| waste_free_multinomial | blocked_backend_correctness | 38 |

## Methods and digests

Contract: proposal_covariance_source=weighted_pre_resample_cloud; proposal_scale=2.38^2 / dimension; target_ess=0.5.

Timing is shown only for correctness-eligible cells; no cross-lane comparison is made.

manifest: `2c2f7bcf0ea4f4ebd6f01330444fd4e50262fa886c87b497943d4308163497e5`; plan: `ce573478ea79bd5b8cca7bf2d73c164e1a55ea784342996627c9fe01f55e1ca9`; source: `1b9197da6df6b949078da4be556de0d2a845b9d62a04305fcc0ec4245e18afb9`; lock: `d41c9e77985bdf463773103d49eb65cd6c742a70df4f880bc3013500c78aa213`; raw: `3ff2b1de4eca7f86bdbc51e8f143a586bd079473d71dd5cb9bdbbd2877aef283`; attempts: `664955b89872ffa38ecd81cbe445f8810a7fec0f7cf123b8adf7e10ae61d8d28`
