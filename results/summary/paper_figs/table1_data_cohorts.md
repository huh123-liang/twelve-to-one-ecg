# table1 data cohorts

Sources: results/eda/table1_label_prevalence.csv and results/summary/external_chapman/{record_status,e7_external_t2}.csv. PTB-XL label rows are multi-label prevalence; Chapman fractions use all scanned records.

| cohort       | measure                            |   count |   denominator |   fraction |
|:-------------|:-----------------------------------|--------:|--------------:|-----------:|
| PTB-XL train | all records                        |   17084 |         17084 |     1      |
| PTB-XL train | NORM                               |    7596 |         17084 |     0.4446 |
| PTB-XL train | MI                                 |    4379 |         17084 |     0.2563 |
| PTB-XL train | STTC                               |    4186 |         17084 |     0.245  |
| PTB-XL train | CD                                 |    3907 |         17084 |     0.2287 |
| PTB-XL train | HYP                                |    2119 |         17084 |     0.124  |
| PTB-XL val   | all records                        |    2146 |          2146 |     1      |
| PTB-XL val   | NORM                               |     955 |          2146 |     0.445  |
| PTB-XL val   | MI                                 |     540 |          2146 |     0.2516 |
| PTB-XL val   | STTC                               |     528 |          2146 |     0.246  |
| PTB-XL val   | CD                                 |     495 |          2146 |     0.2307 |
| PTB-XL val   | HYP                                |     268 |          2146 |     0.1249 |
| PTB-XL test  | all records                        |    2158 |          2158 |     1      |
| PTB-XL test  | NORM                               |     963 |          2158 |     0.4462 |
| PTB-XL test  | MI                                 |     550 |          2158 |     0.2549 |
| PTB-XL test  | STTC                               |     521 |          2158 |     0.2414 |
| PTB-XL test  | CD                                 |     496 |          2158 |     0.2298 |
| PTB-XL test  | HYP                                |     262 |          2158 |     0.1214 |
| Chapman v2   | all scanned                        |   45152 |         45152 |     1      |
| Chapman v2   | included after signal checks       |   34815 |         45152 |     0.7711 |
| Chapman v2   | excluded: diagnosis outside target |    8971 |         45152 |     0.1987 |
| Chapman v2   | excluded: flatline lead            |    1290 |         45152 |     0.0286 |
| Chapman v2   | excluded: nonfinite signal         |      75 |         45152 |     0.0017 |
| Chapman v2   | excluded: header date parse error  |       1 |         45152 |     0      |
| Chapman v2   | included positive                  |   16092 |         45152 |     0.3564 |
| Chapman v2   | included negative                  |   18723 |         45152 |     0.4147 |
