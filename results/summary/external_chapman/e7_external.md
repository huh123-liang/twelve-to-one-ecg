# E7: external Chapman-Shaoxing/Ningbo validation (chapman; v2 mapping)

Records scanned: 45152; included: 34815; external positives: 16092; external negatives: 18723.

The selected SNOMED mapping excludes arrhythmia-only and other unmapped records. This is a restricted T2 triage target, not all possible cardiac abnormalities.

Signal/cache source: `D:\datasets\ecg-arrhythmia\external_chapman_100hz_v2_full.npz`. All 36 checkpoints were frozen PTB-XL runs (three seeds per row).

## Record status

| status                                                         |   records |
|:---------------------------------------------------------------|----------:|
| excluded_signal_flatline_lead                                  |      1290 |
| excluded_signal_nonfinite_signal                               |        75 |
| excluded_signal_time_data_'/'_does_not_match_format_'%d/%m/%Y' |         1 |
| excluded_unmapped_dx                                           |      8971 |
| included                                                       |     34815 |

## T2 triage including scores derived from T1 NORM

| model    | source_task   | leads   |   n_seeds |   external_n |   external_positive |   external_auroc |   external_ci_low |   external_ci_high |   external_auprc |   external_sens_at_spec90 |   external_ece |   ptb_val_spec90_threshold |   external_sens_at_ptb_threshold |   external_spec_at_ptb_threshold |   internal_ptb_test_auroc |   internal_ptb_test_sens_at_spec90 |   delta_auroc_external_minus_internal |   delta_sens_at_spec90_external_minus_internal |
|:---------|:--------------|:--------|----------:|-------------:|--------------------:|-----------------:|------------------:|-------------------:|-----------------:|--------------------------:|---------------:|---------------------------:|---------------------------------:|---------------------------------:|--------------------------:|-----------------------------------:|--------------------------------------:|-----------------------------------------------:|
| resnet1d | t2            | 12lead  |         3 |        34815 |               16092 |           0.9552 |            0.9531 |             0.9572 |           0.9582 |                    0.8815 |         0.0589 |                     0.5132 |                           0.9054 |                           0.8636 |                    0.9401 |                             0.8315 |                                0.0152 |                                         0.05   |
| resnet1d | t2            | leadI   |         3 |        34815 |               16092 |           0.9159 |            0.9126 |             0.9189 |           0.9255 |                    0.7992 |         0.0704 |                     0.5755 |                           0.7974 |                           0.9026 |                    0.8777 |                             0.7055 |                                0.0382 |                                         0.0938 |
| laresnet | t2            | 12lead  |         3 |        34815 |               16092 |           0.9568 |            0.9547 |             0.9588 |           0.959  |                    0.8867 |         0.0525 |                     0.475  |                           0.9102 |                           0.858  |                    0.9408 |                             0.8331 |                                0.0161 |                                         0.0536 |
| laresnet | t2            | limb6   |         3 |        34815 |               16092 |           0.9398 |            0.9371 |             0.9424 |           0.9469 |                    0.8591 |         0.0468 |                     0.5104 |                           0.8699 |                           0.8826 |                    0.9156 |                             0.7825 |                                0.0242 |                                         0.0766 |
| laresnet | t2            | chest6  |         3 |        34815 |               16092 |           0.9482 |            0.9459 |             0.9504 |           0.9521 |                    0.8634 |         0.0905 |                     0.5391 |                           0.9014 |                           0.8399 |                    0.9187 |                             0.7913 |                                0.0295 |                                         0.0721 |
| laresnet | t2            | 2lead   |         3 |        34815 |               16092 |           0.9415 |            0.9387 |             0.9439 |           0.9489 |                    0.8617 |         0.0358 |                     0.4615 |                           0.8706 |                           0.8869 |                    0.9159 |                             0.7841 |                                0.0256 |                                         0.0776 |
| laresnet | t2            | leadII  |         3 |        34815 |               16092 |           0.925  |            0.9222 |             0.9277 |           0.9329 |                    0.8199 |         0.0848 |                     0.5825 |                           0.8368 |                           0.8793 |                    0.8874 |                             0.7319 |                                0.0376 |                                         0.088  |
| laresnet | t2            | leadI   |         3 |        34815 |               16092 |           0.9115 |            0.9082 |             0.9147 |           0.92   |                    0.7911 |         0.0471 |                     0.5007 |                           0.8105 |                           0.8804 |                    0.8754 |                             0.7079 |                                0.0361 |                                         0.0832 |
| resnet1d | t1            | 12lead  |         3 |        34815 |               16092 |           0.9551 |            0.9529 |             0.9572 |           0.9585 |                    0.8859 |       nan      |                     0.4463 |                           0.9122 |                           0.8503 |                    0.9424 |                             0.8347 |                                0.0127 |                                         0.0512 |
| resnet1d | t1            | leadI   |         3 |        34815 |               16092 |           0.9191 |            0.9158 |             0.9221 |           0.9291 |                    0.8081 |       nan      |                     0.4979 |                           0.8047 |                           0.9035 |                    0.8778 |                             0.7111 |                                0.0413 |                                         0.097  |
| bilstm   | t1            | 12lead  |         3 |        34815 |               16092 |           0.9553 |            0.9532 |             0.9574 |           0.9593 |                    0.881  |       nan      |                     0.4516 |                           0.9007 |                           0.8682 |                    0.9405 |                             0.8387 |                                0.0147 |                                         0.0423 |
| bilstm   | t1            | leadI   |         3 |        34815 |               16092 |           0.9183 |            0.9151 |             0.9212 |           0.928  |                    0.8079 |       nan      |                     0.4978 |                           0.8113 |                           0.8966 |                    0.8785 |                             0.7095 |                                0.0398 |                                         0.0984 |

The threshold in each row was selected on PTB-XL fold 9 to reach at least 90% specificity and was applied unchanged to the external set. External AUROC CI uses 1,000 record bootstraps. External sensitivity at 90% specificity is a separate descriptive ROC quantity.

## T1 mapped classes with at least 30 external positives

| model    | leads   | class   |   positive_n |   external_n |   external_auroc |   ci_low |   ci_high |
|:---------|:--------|:--------|-------------:|-------------:|-----------------:|---------:|----------:|
| resnet1d | 12lead  | MI      |          204 |        34815 |           0.957  |   0.944  |    0.9684 |
| resnet1d | 12lead  | STTC    |        13166 |        34815 |           0.9158 |   0.9129 |    0.9188 |
| resnet1d | 12lead  | CD      |         5043 |        34815 |           0.9122 |   0.9078 |    0.9167 |
| resnet1d | 12lead  | HYP     |          757 |        34815 |           0.9372 |   0.9293 |    0.9442 |
| resnet1d | leadI   | MI      |          204 |        34815 |           0.8709 |   0.8464 |    0.8906 |
| resnet1d | leadI   | STTC    |        13166 |        34815 |           0.872  |   0.8677 |    0.876  |
| resnet1d | leadI   | CD      |         5043 |        34815 |           0.8905 |   0.8853 |    0.8957 |
| resnet1d | leadI   | HYP     |          757 |        34815 |           0.8906 |   0.8805 |    0.9006 |
| bilstm   | 12lead  | MI      |          204 |        34815 |           0.9586 |   0.946  |    0.9685 |
| bilstm   | 12lead  | STTC    |        13166 |        34815 |           0.9079 |   0.9047 |    0.911  |
| bilstm   | 12lead  | CD      |         5043 |        34815 |           0.9246 |   0.9208 |    0.9284 |
| bilstm   | 12lead  | HYP     |          757 |        34815 |           0.9419 |   0.9347 |    0.9486 |
| bilstm   | leadI   | MI      |          204 |        34815 |           0.8608 |   0.8354 |    0.8823 |
| bilstm   | leadI   | STTC    |        13166 |        34815 |           0.8776 |   0.8733 |    0.8816 |
| bilstm   | leadI   | CD      |         5043 |        34815 |           0.8996 |   0.8946 |    0.9044 |
| bilstm   | leadI   | HYP     |          757 |        34815 |           0.9032 |   0.894  |    0.9122 |

Wall time: 635.5 s. Figure: `results\summary\fig11_external_chapman.png`.

## Mapping sensitivity comparison

| mapping   |   included |   positive |   negative |   ResNet1d 12lead AUROC |   ResNet1d 12lead sens@spec90 |   ResNet1d leadI AUROC |   ResNet1d leadI sens@spec90 |   LA-ResNet 12lead AUROC |   LA-ResNet 12lead sens@spec90 |   LA-ResNet leadI AUROC |   LA-ResNet leadI sens@spec90 |
|:----------|-----------:|-----------:|-----------:|------------------------:|------------------------------:|-----------------------:|-----------------------------:|-------------------------:|-------------------------------:|------------------------:|------------------------------:|
| v1        |      31364 |      12641 |      18723 |                  0.9533 |                        0.8771 |                 0.91   |                       0.7902 |                   0.9554 |                         0.8825 |                  0.9066 |                        0.7845 |
| v2        |      34815 |      16092 |      18723 |                  0.9552 |                        0.8815 |                 0.9159 |                       0.7992 |                   0.9568 |                         0.8867 |                  0.9115 |                        0.7911 |
| v2_lvhv   |      37313 |      18590 |      18723 |                  0.9363 |                        0.8193 |                 0.8803 |                       0.7301 |                   0.9381 |                         0.829  |                  0.8752 |                        0.7214 |

v2 is the main analysis; v1 is the archived initial mapping, and v2_lvhv additionally treats left ventricular high voltage as HYP. PTB-XL internal test results and operating-threshold rules are identical across these rows.
