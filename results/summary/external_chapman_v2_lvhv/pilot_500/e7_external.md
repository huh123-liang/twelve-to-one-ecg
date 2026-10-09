# E7: external Chapman-Shaoxing/Ningbo validation (chapman; v2_lvhv mapping)

Records scanned: 500; included: 440; external positives: 216; external negatives: 224.

The selected SNOMED mapping excludes arrhythmia-only and other unmapped records. This is a restricted T2 triage target, not all possible cardiac abnormalities.

Signal/cache source: `D:\datasets\ecg-arrhythmia\external_chapman_100hz_v2_first500.npz plus newly mapped LVHV signals from D:\datasets\ecg-arrhythmia\WFDBRecords`. All 36 checkpoints were frozen PTB-XL runs (three seeds per row).

## Record status

| status               |   records |
|:---------------------|----------:|
| excluded_unmapped_dx |        60 |
| included             |       440 |

## T2 triage including scores derived from T1 NORM

| model    | source_task   | leads   |   n_seeds |   external_n |   external_positive |   external_auroc |   external_ci_low |   external_ci_high |   external_auprc |   external_sens_at_spec90 |   external_ece |   ptb_val_spec90_threshold |   external_sens_at_ptb_threshold |   external_spec_at_ptb_threshold |   internal_ptb_test_auroc |   internal_ptb_test_sens_at_spec90 |   delta_auroc_external_minus_internal |   delta_sens_at_spec90_external_minus_internal |
|:---------|:--------------|:--------|----------:|-------------:|--------------------:|-----------------:|------------------:|-------------------:|-----------------:|--------------------------:|---------------:|---------------------------:|---------------------------------:|---------------------------------:|--------------------------:|-----------------------------------:|--------------------------------------:|-----------------------------------------------:|
| resnet1d | t2            | 12lead  |         3 |          440 |                 216 |           0.9365 |            0.9156 |             0.9573 |           0.941  |                    0.7963 |         0.0481 |                     0.5132 |                           0.8009 |                           0.8973 |                    0.9401 |                             0.8315 |                               -0.0035 |                                        -0.0352 |
| resnet1d | t2            | leadI   |         3 |          440 |                 216 |           0.8856 |            0.852  |             0.9169 |           0.9123 |                    0.7315 |         0.045  |                     0.5755 |                           0.7083 |                           0.9241 |                    0.8777 |                             0.7055 |                                0.0079 |                                         0.026  |
| laresnet | t2            | 12lead  |         3 |          440 |                 216 |           0.9392 |            0.9177 |             0.9594 |           0.9444 |                    0.8102 |         0.0664 |                     0.475  |                           0.8102 |                           0.8973 |                    0.9408 |                             0.8331 |                               -0.0016 |                                        -0.0229 |
| laresnet | t2            | limb6   |         3 |          440 |                 216 |           0.8912 |            0.8558 |             0.9218 |           0.9191 |                    0.7593 |         0.0543 |                     0.5104 |                           0.7407 |                           0.9241 |                    0.9156 |                             0.7825 |                               -0.0244 |                                        -0.0232 |
| laresnet | t2            | chest6  |         3 |          440 |                 216 |           0.946  |            0.9253 |             0.9634 |           0.9478 |                    0.8241 |         0.0577 |                     0.5391 |                           0.8333 |                           0.8839 |                    0.9187 |                             0.7913 |                                0.0273 |                                         0.0327 |
| laresnet | t2            | 2lead   |         3 |          440 |                 216 |           0.8967 |            0.8641 |             0.9283 |           0.9243 |                    0.7731 |         0.0571 |                     0.4615 |                           0.75   |                           0.9375 |                    0.9159 |                             0.7841 |                               -0.0192 |                                        -0.011  |
| laresnet | t2            | leadII  |         3 |          440 |                 216 |           0.8828 |            0.8458 |             0.9152 |           0.9101 |                    0.7269 |         0.0405 |                     0.5825 |                           0.6991 |                           0.9375 |                    0.8874 |                             0.7319 |                               -0.0045 |                                        -0.0051 |
| laresnet | t2            | leadI   |         3 |          440 |                 216 |           0.8735 |            0.8371 |             0.9051 |           0.9032 |                    0.7269 |         0.0665 |                     0.5007 |                           0.6944 |                           0.9196 |                    0.8754 |                             0.7079 |                               -0.0019 |                                         0.019  |
| resnet1d | t1            | 12lead  |         3 |          440 |                 216 |           0.9464 |            0.9266 |             0.9644 |           0.9503 |                    0.8287 |       nan      |                     0.4463 |                           0.8287 |                           0.8929 |                    0.9424 |                             0.8347 |                                0.004  |                                        -0.006  |
| resnet1d | t1            | leadI   |         3 |          440 |                 216 |           0.8918 |            0.8598 |             0.9215 |           0.916  |                    0.7546 |       nan      |                     0.4979 |                           0.7083 |                           0.933  |                    0.8778 |                             0.7111 |                                0.014  |                                         0.0436 |
| bilstm   | t1            | 12lead  |         3 |          440 |                 216 |           0.9447 |            0.9249 |             0.9626 |           0.9485 |                    0.8426 |       nan      |                     0.4516 |                           0.8102 |                           0.9196 |                    0.9405 |                             0.8387 |                                0.0041 |                                         0.0039 |
| bilstm   | t1            | leadI   |         3 |          440 |                 216 |           0.8857 |            0.853  |             0.9172 |           0.9122 |                    0.7593 |       nan      |                     0.4978 |                           0.6991 |                           0.9152 |                    0.8785 |                             0.7095 |                                0.0072 |                                         0.0498 |

The threshold in each row was selected on PTB-XL fold 9 to reach at least 90% specificity and was applied unchanged to the external set. External AUROC CI uses 1,000 record bootstraps. External sensitivity at 90% specificity is a separate descriptive ROC quantity.

## T1 mapped classes with at least 30 external positives

| model    | leads   | class   |   positive_n |   external_n |   external_auroc |   ci_low |   ci_high |
|:---------|:--------|:--------|-------------:|-------------:|-----------------:|---------:|----------:|
| resnet1d | 12lead  | STTC    |          127 |          440 |           0.936  |   0.9103 |    0.9572 |
| resnet1d | 12lead  | CD      |           68 |          440 |           0.9014 |   0.8606 |    0.9376 |
| resnet1d | 12lead  | HYP     |           74 |          440 |           0.9032 |   0.8724 |    0.9318 |
| resnet1d | leadI   | STTC    |          127 |          440 |           0.9103 |   0.8743 |    0.941  |
| resnet1d | leadI   | CD      |           68 |          440 |           0.9043 |   0.8598 |    0.939  |
| resnet1d | leadI   | HYP     |           74 |          440 |           0.6747 |   0.6154 |    0.7358 |
| bilstm   | 12lead  | STTC    |          127 |          440 |           0.939  |   0.9104 |    0.9601 |
| bilstm   | 12lead  | CD      |           68 |          440 |           0.9275 |   0.8954 |    0.9545 |
| bilstm   | 12lead  | HYP     |           74 |          440 |           0.901  |   0.8667 |    0.9313 |
| bilstm   | leadI   | STTC    |          127 |          440 |           0.9031 |   0.8654 |    0.9338 |
| bilstm   | leadI   | CD      |           68 |          440 |           0.9108 |   0.8716 |    0.9419 |
| bilstm   | leadI   | HYP     |           74 |          440 |           0.6709 |   0.6112 |    0.7352 |

Wall time: 15.1 s. Figure: `results\summary\fig11_external_chapman_v2_lvhv_pilot.png`.
