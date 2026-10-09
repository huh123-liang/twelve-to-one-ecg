# E7: external Chapman-Shaoxing/Ningbo validation (chapman; v2 mapping)

Records scanned: 500; included: 397; external positives: 173; external negatives: 224.

The selected SNOMED mapping excludes arrhythmia-only and other unmapped records. This is a restricted T2 triage target, not all possible cardiac abnormalities.

Signal/cache source: `D:\datasets\ecg-arrhythmia\external_chapman_100hz_v2_first500.npz`. All 36 checkpoints were frozen PTB-XL runs (three seeds per row).

## Record status

| status               |   records |
|:---------------------|----------:|
| excluded_unmapped_dx |       103 |
| included             |       397 |

## T2 triage including scores derived from T1 NORM

| model    | source_task   | leads   |   n_seeds |   external_n |   external_positive |   external_auroc |   external_ci_low |   external_ci_high |   external_auprc |   external_sens_at_spec90 |   external_ece |   ptb_val_spec90_threshold |   external_sens_at_ptb_threshold |   external_spec_at_ptb_threshold |   internal_ptb_test_auroc |   internal_ptb_test_sens_at_spec90 |   delta_auroc_external_minus_internal |   delta_sens_at_spec90_external_minus_internal |
|:---------|:--------------|:--------|----------:|-------------:|--------------------:|-----------------:|------------------:|-------------------:|-----------------:|--------------------------:|---------------:|---------------------------:|---------------------------------:|---------------------------------:|--------------------------:|-----------------------------------:|--------------------------------------:|-----------------------------------------------:|
| resnet1d | t2            | 12lead  |         3 |          397 |                 173 |           0.9601 |            0.9414 |             0.9774 |           0.9581 |                    0.896  |         0.0566 |                     0.5132 |                           0.9017 |                           0.8973 |                    0.9401 |                             0.8315 |                                0.0201 |                                         0.0645 |
| resnet1d | t2            | leadI   |         3 |          397 |                 173 |           0.9441 |            0.9185 |             0.9673 |           0.9479 |                    0.8439 |         0.0964 |                     0.5755 |                           0.8324 |                           0.9241 |                    0.8777 |                             0.7055 |                                0.0664 |                                         0.1385 |
| laresnet | t2            | 12lead  |         3 |          397 |                 173 |           0.9627 |            0.9452 |             0.9789 |           0.9607 |                    0.9075 |         0.0558 |                     0.475  |                           0.9075 |                           0.8973 |                    0.9408 |                             0.8331 |                                0.0219 |                                         0.0744 |
| laresnet | t2            | limb6   |         3 |          397 |                 173 |           0.9534 |            0.9303 |             0.9738 |           0.9578 |                    0.8786 |         0.046  |                     0.5104 |                           0.8671 |                           0.9241 |                    0.9156 |                             0.7825 |                                0.0378 |                                         0.0961 |
| laresnet | t2            | chest6  |         3 |          397 |                 173 |           0.9638 |            0.9466 |             0.9788 |           0.9599 |                    0.9133 |         0.0831 |                     0.5391 |                           0.9133 |                           0.8839 |                    0.9187 |                             0.7913 |                                0.0451 |                                         0.122  |
| laresnet | t2            | 2lead   |         3 |          397 |                 173 |           0.9598 |            0.9386 |             0.978  |           0.9631 |                    0.896  |         0.0507 |                     0.4615 |                           0.8786 |                           0.9375 |                    0.9159 |                             0.7841 |                                0.0439 |                                         0.1118 |
| laresnet | t2            | leadII  |         3 |          397 |                 173 |           0.9454 |            0.9223 |             0.9659 |           0.9461 |                    0.8382 |         0.0938 |                     0.5825 |                           0.8092 |                           0.9375 |                    0.8874 |                             0.7319 |                                0.058  |                                         0.1062 |
| laresnet | t2            | leadI   |         3 |          397 |                 173 |           0.9346 |            0.9081 |             0.9601 |           0.9384 |                    0.8382 |         0.0768 |                     0.5007 |                           0.815  |                           0.9196 |                    0.8754 |                             0.7079 |                                0.0592 |                                         0.1303 |
| resnet1d | t1            | 12lead  |         3 |          397 |                 173 |           0.9631 |            0.9452 |             0.9795 |           0.9618 |                    0.9017 |       nan      |                     0.4463 |                           0.9017 |                           0.8929 |                    0.9424 |                             0.8347 |                                0.0207 |                                         0.0671 |
| resnet1d | t1            | leadI   |         3 |          397 |                 173 |           0.9449 |            0.9204 |             0.9676 |           0.9485 |                    0.8613 |       nan      |                     0.4979 |                           0.8266 |                           0.933  |                    0.8778 |                             0.7111 |                                0.067  |                                         0.1502 |
| bilstm   | t1            | 12lead  |         3 |          397 |                 173 |           0.9669 |            0.9501 |             0.982  |           0.9644 |                    0.9133 |       nan      |                     0.4516 |                           0.896  |                           0.9196 |                    0.9405 |                             0.8387 |                                0.0263 |                                         0.0746 |
| bilstm   | t1            | leadI   |         3 |          397 |                 173 |           0.9452 |            0.9214 |             0.9682 |           0.9476 |                    0.8613 |       nan      |                     0.4978 |                           0.8092 |                           0.9152 |                    0.8785 |                             0.7095 |                                0.0667 |                                         0.1518 |

The threshold in each row was selected on PTB-XL fold 9 to reach at least 90% specificity and was applied unchanged to the external set. External AUROC CI uses 1,000 record bootstraps. External sensitivity at 90% specificity is a separate descriptive ROC quantity.

## T1 mapped classes with at least 30 external positives

| model    | leads   | class   |   positive_n |   external_n |   external_auroc |   ci_low |   ci_high |
|:---------|:--------|:--------|-------------:|-------------:|-----------------:|---------:|----------:|
| resnet1d | 12lead  | STTC    |          127 |          397 |           0.9441 |   0.9215 |    0.9649 |
| resnet1d | 12lead  | CD      |           68 |          397 |           0.9015 |   0.8601 |    0.9362 |
| resnet1d | leadI   | STTC    |          127 |          397 |           0.9162 |   0.8825 |    0.9465 |
| resnet1d | leadI   | CD      |           68 |          397 |           0.9036 |   0.8597 |    0.9419 |
| bilstm   | 12lead  | STTC    |          127 |          397 |           0.9427 |   0.9163 |    0.9658 |
| bilstm   | 12lead  | CD      |           68 |          397 |           0.9279 |   0.8942 |    0.9572 |
| bilstm   | leadI   | STTC    |          127 |          397 |           0.9091 |   0.8745 |    0.9406 |
| bilstm   | leadI   | CD      |           68 |          397 |           0.9119 |   0.8719 |    0.9443 |

Wall time: 16.5 s. Figure: `results\summary\fig11_external_chapman_pilot.png`.
