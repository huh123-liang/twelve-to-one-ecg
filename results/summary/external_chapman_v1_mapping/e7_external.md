# E7: external Chapman-Shaoxing/Ningbo validation (chapman)

Records scanned: 45152; included: 31364; external positives: 12641; external negatives: 18723.

The fixed SNOMED mapping excludes arrhythmia-only and other unmapped records. This is a restricted T2 triage target, not all possible cardiac abnormalities.

Signal/cache source: `D:\datasets\ecg-arrhythmia\external_chapman_100hz_full.npz`. All 36 checkpoints were frozen PTB-XL runs (three seeds per row).

## Record status

| status                                                         |   records |
|:---------------------------------------------------------------|----------:|
| excluded_signal_flatline_lead                                  |      1230 |
| excluded_signal_nonfinite_signal                               |        67 |
| excluded_signal_time_data_'/'_does_not_match_format_'%d/%m/%Y' |         1 |
| excluded_unmapped_dx                                           |     12490 |
| included                                                       |     31364 |

## T2 triage including scores derived from T1 NORM

| model    | source_task   | leads   |   n_seeds |   external_n |   external_positive |   external_auroc |   external_ci_low |   external_ci_high |   external_auprc |   external_sens_at_spec90 |   external_ece |   ptb_val_spec90_threshold |   external_sens_at_ptb_threshold |   external_spec_at_ptb_threshold |   internal_ptb_test_auroc |   internal_ptb_test_sens_at_spec90 |   delta_auroc_external_minus_internal |   delta_sens_at_spec90_external_minus_internal |
|:---------|:--------------|:--------|----------:|-------------:|--------------------:|-----------------:|------------------:|-------------------:|-----------------:|--------------------------:|---------------:|---------------------------:|---------------------------------:|---------------------------------:|--------------------------:|-----------------------------------:|--------------------------------------:|-----------------------------------------------:|
| resnet1d | t2            | 12lead  |         3 |        31364 |               12641 |           0.9533 |            0.951  |             0.9557 |           0.9477 |                    0.8771 |         0.077  |                     0.5132 |                           0.9006 |                           0.8636 |                    0.9401 |                             0.8315 |                                0.0133 |                                         0.0457 |
| resnet1d | t2            | leadI   |         3 |        31364 |               12641 |           0.91   |            0.9068 |             0.9138 |           0.9064 |                    0.7902 |         0.0956 |                     0.5755 |                           0.7884 |                           0.9026 |                    0.8777 |                             0.7055 |                                0.0323 |                                         0.0847 |
| laresnet | t2            | 12lead  |         3 |        31364 |               12641 |           0.9554 |            0.9531 |             0.9578 |           0.9489 |                    0.8825 |         0.0653 |                     0.475  |                           0.9072 |                           0.858  |                    0.9408 |                             0.8331 |                                0.0146 |                                         0.0495 |
| laresnet | t2            | limb6   |         3 |        31364 |               12641 |           0.9356 |            0.9325 |             0.9387 |           0.9334 |                    0.8518 |         0.0649 |                     0.5104 |                           0.8632 |                           0.8826 |                    0.9156 |                             0.7825 |                                0.02   |                                         0.0693 |
| laresnet | t2            | chest6  |         3 |        31364 |               12641 |           0.947  |            0.9445 |             0.9495 |           0.941  |                    0.8605 |         0.1118 |                     0.5391 |                           0.8993 |                           0.8399 |                    0.9187 |                             0.7913 |                                0.0283 |                                         0.0691 |
| laresnet | t2            | 2lead   |         3 |        31364 |               12641 |           0.938  |            0.9352 |             0.9409 |           0.9364 |                    0.855  |         0.0486 |                     0.4615 |                           0.864  |                           0.8869 |                    0.9159 |                             0.7841 |                                0.0222 |                                         0.0709 |
| laresnet | t2            | leadII  |         3 |        31364 |               12641 |           0.9204 |            0.917  |             0.9239 |           0.9167 |                    0.8133 |         0.11   |                     0.5825 |                           0.8298 |                           0.8793 |                    0.8874 |                             0.7319 |                                0.0331 |                                         0.0814 |
| laresnet | t2            | leadI   |         3 |        31364 |               12641 |           0.9066 |            0.9033 |             0.9105 |           0.9011 |                    0.7845 |         0.0664 |                     0.5007 |                           0.8033 |                           0.8804 |                    0.8754 |                             0.7079 |                                0.0313 |                                         0.0766 |
| resnet1d | t1            | 12lead  |         3 |        31364 |               12641 |           0.9533 |            0.9509 |             0.9557 |           0.9488 |                    0.8828 |       nan      |                     0.4463 |                           0.9088 |                           0.8503 |                    0.9424 |                             0.8347 |                                0.0109 |                                         0.0482 |
| resnet1d | t1            | leadI   |         3 |        31364 |               12641 |           0.9146 |            0.9115 |             0.9181 |           0.9119 |                    0.8021 |       nan      |                     0.4979 |                           0.7984 |                           0.9035 |                    0.8778 |                             0.7111 |                                0.0368 |                                         0.091  |
| bilstm   | t1            | 12lead  |         3 |        31364 |               12641 |           0.9532 |            0.9508 |             0.9556 |           0.9494 |                    0.8771 |       nan      |                     0.4516 |                           0.8976 |                           0.8682 |                    0.9405 |                             0.8387 |                                0.0126 |                                         0.0385 |
| bilstm   | t1            | leadI   |         3 |        31364 |               12641 |           0.914  |            0.9107 |             0.9176 |           0.9106 |                    0.8007 |       nan      |                     0.4978 |                           0.8044 |                           0.8966 |                    0.8785 |                             0.7095 |                                0.0355 |                                         0.0913 |

The threshold in each row was selected on PTB-XL fold 9 to reach at least 90% specificity and was applied unchanged to the external set. External AUROC CI uses 1,000 record bootstraps. External sensitivity at 90% specificity is a separate descriptive ROC quantity.

## T1 mapped classes with at least 30 external positives

| model    | leads   | class   |   positive_n |   external_n |   external_auroc |   ci_low |   ci_high |
|:---------|:--------|:--------|-------------:|-------------:|-----------------:|---------:|----------:|
| resnet1d | 12lead  | MI      |          122 |        31364 |           0.9434 |   0.9242 |    0.9616 |
| resnet1d | 12lead  | STTC    |        10226 |        31364 |           0.8985 |   0.8949 |    0.9025 |
| resnet1d | 12lead  | CD      |         3141 |        31364 |           0.8631 |   0.8571 |    0.8694 |
| resnet1d | 12lead  | HYP     |          756 |        31364 |           0.9452 |   0.9385 |    0.9519 |
| resnet1d | leadI   | MI      |          122 |        31364 |           0.8674 |   0.8443 |    0.8916 |
| resnet1d | leadI   | STTC    |        10226 |        31364 |           0.8564 |   0.8516 |    0.8614 |
| resnet1d | leadI   | CD      |         3141 |        31364 |           0.8526 |   0.8458 |    0.8592 |
| resnet1d | leadI   | HYP     |          756 |        31364 |           0.9036 |   0.8949 |    0.9121 |
| bilstm   | 12lead  | MI      |          122 |        31364 |           0.9471 |   0.9301 |    0.9623 |
| bilstm   | 12lead  | STTC    |        10226 |        31364 |           0.8887 |   0.885  |    0.8929 |
| bilstm   | 12lead  | CD      |         3141 |        31364 |           0.8816 |   0.8765 |    0.8873 |
| bilstm   | 12lead  | HYP     |          756 |        31364 |           0.9495 |   0.9425 |    0.9552 |
| bilstm   | leadI   | MI      |          122 |        31364 |           0.8578 |   0.8288 |    0.8868 |
| bilstm   | leadI   | STTC    |        10226 |        31364 |           0.8601 |   0.8556 |    0.8651 |
| bilstm   | leadI   | CD      |         3141 |        31364 |           0.868  |   0.8622 |    0.8741 |
| bilstm   | leadI   | HYP     |          756 |        31364 |           0.9158 |   0.9077 |    0.9236 |

Wall time: 1123.3 s. Figure: `results\summary\fig11_external_chapman.png`.
