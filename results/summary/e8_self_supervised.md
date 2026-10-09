# E8: masked-signal pretraining with 10% labeled PTB-XL data

The two ECG-ViT encoders were pretrained without labels on folds 1-8; fold 9 selected the reconstruction checkpoint. Both supervised methods use the same seed-specific 10% patient subset.

## Reconstruction pretraining

| leads   |   best_val_mse |   zero_prediction_mse |   ratio_to_zero |   best_epoch |   epochs_run |   train_time_s |
|:--------|---------------:|----------------------:|----------------:|-------------:|-------------:|---------------:|
| 12lead  |        0.2182  |               1.00199 |         0.21777 |           98 |          100 |        1806.03 |
| leadI   |        0.18547 |               0.89078 |         0.20821 |           96 |          100 |        1344.36 |

## Three-seed classification results

| leads   | method   |   train_fraction |   n_seeds |   seed_mean_auroc |   seed_std_auroc |   ensemble_auroc |   ensemble_ci_low |   ensemble_ci_high |   full_data_vit_reference_auroc |   seed0_auroc |   seed1_auroc |   seed2_auroc |
|:--------|:---------|-----------------:|----------:|------------------:|-----------------:|-----------------:|------------------:|-------------------:|--------------------------------:|--------------:|--------------:|--------------:|
| 12lead  | scratch  |              0.1 |         3 |           0.86679 |          0.00779 |          0.88596 |           0.87709 |            0.89546 |                         0.92209 |       0.87562 |       0.86089 |       0.86385 |
| 12lead  | msm      |              0.1 |         3 |           0.87594 |          0.00057 |          0.89243 |           0.8837  |            0.90121 |                         0.92209 |       0.87652 |       0.87594 |       0.87537 |
| leadI   | scratch  |              0.1 |         3 |           0.79194 |          0.00264 |          0.80535 |           0.79378 |            0.81846 |                         0.82871 |       0.79336 |       0.79357 |       0.78889 |
| leadI   | msm      |              0.1 |         3 |           0.80965 |          0.00049 |          0.82029 |           0.8087  |            0.83184 |                         0.82871 |       0.81021 |       0.80929 |       0.80944 |

## Paired test on seed-ensemble predictions

| leads   | comparison        |    diff |   ci95_low |   ci95_high |   p_value |   seed0_diff |   seed1_diff |   seed2_diff |
|:--------|:------------------|--------:|-----------:|------------:|----------:|-------------:|-------------:|-------------:|
| 12lead  | msm_minus_scratch | 0.00647 |    0.00198 |     0.0112  |     0.006 |      0.00089 |      0.01506 |      0.01152 |
| leadI   | msm_minus_scratch | 0.01494 |    0.0091  |     0.02099 |     0     |      0.01685 |      0.01573 |      0.02056 |

The difference is MSM minus scratch. The 95% CI and two-sided p value use 1,000 paired record bootstraps; a displayed zero p value means below bootstrap resolution, not an exact probability of zero.

The full-data ViT reference comes from the already completed E1/E2 runs and was not retrained.
