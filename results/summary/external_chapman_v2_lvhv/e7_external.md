# E7: external Chapman-Shaoxing/Ningbo validation (chapman; v2_lvhv mapping)

Records scanned: 45152; included: 37313; external positives: 18590; external negatives: 18723.

The selected SNOMED mapping excludes arrhythmia-only and other unmapped records. This is a restricted T2 triage target, not all possible cardiac abnormalities.

Signal/cache source: `D:\datasets\ecg-arrhythmia\external_chapman_100hz_v2_full.npz plus newly mapped LVHV signals from D:\datasets\ecg-arrhythmia\WFDBRecords`. All 36 checkpoints were frozen PTB-XL runs (three seeds per row).

## Record status

| status                                                         |   records |
|:---------------------------------------------------------------|----------:|
| excluded_signal_flatline_lead                                  |      1299 |
| excluded_signal_nonfinite_signal                               |        86 |
| excluded_signal_time_data_'/'_does_not_match_format_'%d/%m/%Y' |         1 |
| excluded_unmapped_dx                                           |      6453 |
| included                                                       |     37313 |

## T2 triage including scores derived from T1 NORM

| model    | source_task   | leads   |   n_seeds |   external_n |   external_positive |   external_auroc |   external_ci_low |   external_ci_high |   external_auprc |   external_sens_at_spec90 |   external_ece |   ptb_val_spec90_threshold |   external_sens_at_ptb_threshold |   external_spec_at_ptb_threshold |   internal_ptb_test_auroc |   internal_ptb_test_sens_at_spec90 |   delta_auroc_external_minus_internal |   delta_sens_at_spec90_external_minus_internal |
|:---------|:--------------|:--------|----------:|-------------:|--------------------:|-----------------:|------------------:|-------------------:|-----------------:|--------------------------:|---------------:|---------------------------:|---------------------------------:|---------------------------------:|--------------------------:|-----------------------------------:|--------------------------------------:|-----------------------------------------------:|
| resnet1d | t2            | 12lead  |         3 |        37313 |               18590 |           0.9363 |            0.9339 |             0.9387 |           0.9458 |                    0.8193 |         0.0289 |                     0.5132 |                           0.8506 |                           0.8636 |                    0.9401 |                             0.8315 |                               -0.0037 |                                        -0.0121 |
| resnet1d | t2            | leadI   |         3 |        37313 |               18590 |           0.8803 |            0.877  |             0.8839 |           0.9061 |                    0.7301 |         0.0323 |                     0.5755 |                           0.7282 |                           0.9026 |                    0.8777 |                             0.7055 |                                0.0026 |                                         0.0247 |
| laresnet | t2            | 12lead  |         3 |        37313 |               18590 |           0.9381 |            0.9359 |             0.9404 |           0.9472 |                    0.829  |         0.0452 |                     0.475  |                           0.8603 |                           0.858  |                    0.9408 |                             0.8331 |                               -0.0027 |                                        -0.0041 |
| laresnet | t2            | limb6   |         3 |        37313 |               18590 |           0.902  |            0.8988 |             0.9051 |           0.925  |                    0.7865 |         0.041  |                     0.5104 |                           0.799  |                           0.8826 |                    0.9156 |                             0.7825 |                               -0.0136 |                                         0.004  |
| laresnet | t2            | chest6  |         3 |        37313 |               18590 |           0.9334 |            0.9311 |             0.9359 |           0.9432 |                    0.8107 |         0.0597 |                     0.5391 |                           0.8626 |                           0.8399 |                    0.9187 |                             0.7913 |                                0.0147 |                                         0.0194 |
| laresnet | t2            | 2lead   |         3 |        37313 |               18590 |           0.9014 |            0.8982 |             0.9045 |           0.9257 |                    0.7874 |         0.0351 |                     0.4615 |                           0.7971 |                           0.8869 |                    0.9159 |                             0.7841 |                               -0.0144 |                                         0.0033 |
| laresnet | t2            | leadII  |         3 |        37313 |               18590 |           0.8849 |            0.8814 |             0.8881 |           0.9105 |                    0.7458 |         0.0478 |                     0.5825 |                           0.7641 |                           0.8793 |                    0.8874 |                             0.7319 |                               -0.0025 |                                         0.0139 |
| laresnet | t2            | leadI   |         3 |        37313 |               18590 |           0.8752 |            0.8717 |             0.8789 |           0.9007 |                    0.7214 |         0.0247 |                     0.5007 |                           0.7412 |                           0.8804 |                    0.8754 |                             0.7079 |                               -0.0002 |                                         0.0135 |
| resnet1d | t1            | 12lead  |         3 |        37313 |               18590 |           0.9414 |            0.9392 |             0.9435 |           0.9496 |                    0.8335 |       nan      |                     0.4463 |                           0.8742 |                           0.8503 |                    0.9424 |                             0.8347 |                               -0.001  |                                        -0.0012 |
| resnet1d | t1            | leadI   |         3 |        37313 |               18590 |           0.8874 |            0.8842 |             0.8909 |           0.9117 |                    0.7425 |       nan      |                     0.4979 |                           0.7389 |                           0.9035 |                    0.8778 |                             0.7111 |                                0.0096 |                                         0.0315 |
| bilstm   | t1            | 12lead  |         3 |        37313 |               18590 |           0.9386 |            0.9364 |             0.9407 |           0.9484 |                    0.8235 |       nan      |                     0.4516 |                           0.8507 |                           0.8682 |                    0.9405 |                             0.8387 |                               -0.0019 |                                        -0.0152 |
| bilstm   | t1            | leadI   |         3 |        37313 |               18590 |           0.8838 |            0.8805 |             0.8873 |           0.909  |                    0.7381 |       nan      |                     0.4978 |                           0.7421 |                           0.8966 |                    0.8785 |                             0.7095 |                                0.0053 |                                         0.0286 |

The threshold in each row was selected on PTB-XL fold 9 to reach at least 90% specificity and was applied unchanged to the external set. External AUROC CI uses 1,000 record bootstraps. External sensitivity at 90% specificity is a separate descriptive ROC quantity.

## T1 mapped classes with at least 30 external positives

| model    | leads   | class   |   positive_n |   external_n |   external_auroc |   ci_low |   ci_high |
|:---------|:--------|:--------|-------------:|-------------:|-----------------:|---------:|----------:|
| resnet1d | 12lead  | MI      |          204 |        37313 |           0.9588 |   0.9469 |    0.9694 |
| resnet1d | 12lead  | STTC    |        13166 |        37313 |           0.908  |   0.905  |    0.9111 |
| resnet1d | 12lead  | CD      |         5043 |        37313 |           0.9123 |   0.9079 |    0.9166 |
| resnet1d | 12lead  | HYP     |         6115 |        37313 |           0.9273 |   0.9242 |    0.9302 |
| resnet1d | leadI   | MI      |          204 |        37313 |           0.8752 |   0.8543 |    0.8965 |
| resnet1d | leadI   | STTC    |        13166 |        37313 |           0.8634 |   0.8593 |    0.8678 |
| resnet1d | leadI   | CD      |         5043 |        37313 |           0.8916 |   0.8866 |    0.8964 |
| resnet1d | leadI   | HYP     |         6115 |        37313 |           0.7144 |   0.7082 |    0.7209 |
| bilstm   | 12lead  | MI      |          204 |        37313 |           0.9604 |   0.9491 |    0.9702 |
| bilstm   | 12lead  | STTC    |        13166 |        37313 |           0.9021 |   0.899  |    0.9054 |
| bilstm   | 12lead  | CD      |         5043 |        37313 |           0.9232 |   0.9196 |    0.9271 |
| bilstm   | 12lead  | HYP     |         6115 |        37313 |           0.9186 |   0.9152 |    0.9217 |
| bilstm   | leadI   | MI      |          204 |        37313 |           0.8656 |   0.8408 |    0.8895 |
| bilstm   | leadI   | STTC    |        13166 |        37313 |           0.8695 |   0.8653 |    0.8737 |
| bilstm   | leadI   | CD      |         5043 |        37313 |           0.9004 |   0.8956 |    0.905  |
| bilstm   | leadI   | HYP     |         6115 |        37313 |           0.7089 |   0.7024 |    0.7153 |

Wall time: 418.2 s. Figure: `results\summary\fig11_external_chapman_v2_lvhv.png`.

## Mapping sensitivity comparison

| mapping   |   included |   positive |   negative |   ResNet1d 12lead AUROC |   ResNet1d 12lead sens@spec90 |   ResNet1d leadI AUROC |   ResNet1d leadI sens@spec90 |   LA-ResNet 12lead AUROC |   LA-ResNet 12lead sens@spec90 |   LA-ResNet leadI AUROC |   LA-ResNet leadI sens@spec90 |
|:----------|-----------:|-----------:|-----------:|------------------------:|------------------------------:|-----------------------:|-----------------------------:|-------------------------:|-------------------------------:|------------------------:|------------------------------:|
| v1        |      31364 |      12641 |      18723 |                  0.9533 |                        0.8771 |                 0.91   |                       0.7902 |                   0.9554 |                         0.8825 |                  0.9066 |                        0.7845 |
| v2        |      34815 |      16092 |      18723 |                  0.9552 |                        0.8815 |                 0.9159 |                       0.7992 |                   0.9568 |                         0.8867 |                  0.9115 |                        0.7911 |
| v2_lvhv   |      37313 |      18590 |      18723 |                  0.9363 |                        0.8193 |                 0.8803 |                       0.7301 |                   0.9381 |                         0.829  |                  0.8752 |                        0.7214 |

v2 is the main analysis; v1 is the archived initial mapping, and v2_lvhv additionally treats left ventricular high voltage as HYP. PTB-XL internal test results and operating-threshold rules are identical across these rows.
