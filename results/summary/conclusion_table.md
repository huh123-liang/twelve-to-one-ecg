# Conclusion table

Criterion (pre-specified): screen = AUROC drop non-inferior (95% CI lower bound of Δ ≥ −0.02) AND sensitivity ≥ 0.80 at 90% specificity; conditional = one of the two; otherwise not replaceable.
Each cell: verdict · ΔAUROC [95% CI] · sens@spec90. Predictions averaged over seeds.

## Task T1 — primary approach (chosen on validation): bilstm [S1_scratch]

| class | limb6 | chest6 | 2lead | leadII | leadI |
|---|---|---|---|---|---|
| NORM | ✘ not replaceable · -0.020 [-0.027, -0.014] · 0.73 | ✘ not replaceable · -0.020 [-0.026, -0.013] · 0.72 | ✘ not replaceable · -0.019 [-0.025, -0.014] · 0.73 | ✘ not replaceable · -0.045 [-0.055, -0.037] · 0.66 | ✘ not replaceable · -0.058 [-0.070, -0.048] · 0.60 |
| MI | ✘ not replaceable · -0.041 [-0.052, -0.029] · 0.67 | ✘ not replaceable · -0.048 [-0.062, -0.033] · 0.66 | ✘ not replaceable · -0.042 [-0.053, -0.030] · 0.65 | ✘ not replaceable · -0.099 [-0.115, -0.084] · 0.51 | ✘ not replaceable · -0.128 [-0.147, -0.107] · 0.45 |
| STTC | ✘ not replaceable · -0.020 [-0.027, -0.012] · 0.73 | ✔ screen · -0.004 [-0.010, +0.001] · 0.84 | ✘ not replaceable · -0.018 [-0.026, -0.010] · 0.74 | ✘ not replaceable · -0.058 [-0.071, -0.046] · 0.64 | ✘ not replaceable · -0.059 [-0.072, -0.045] · 0.64 |
| CD | ✘ not replaceable · -0.033 [-0.045, -0.022] · 0.70 | ✘ not replaceable · -0.024 [-0.033, -0.016] · 0.73 | ✘ not replaceable · -0.034 [-0.047, -0.022] · 0.72 | ✘ not replaceable · -0.053 [-0.067, -0.039] · 0.70 | ✘ not replaceable · -0.104 [-0.125, -0.082] · 0.59 |
| HYP | ✘ not replaceable · -0.049 [-0.066, -0.034] · 0.58 | ✘ not replaceable · -0.023 [-0.037, -0.009] · 0.69 | ✘ not replaceable · -0.048 [-0.063, -0.032] · 0.60 | ✘ not replaceable · -0.104 [-0.128, -0.078] · 0.47 | ✘ not replaceable · -0.093 [-0.112, -0.073] · 0.52 |
| ABNORMAL (derived 1-P(NORM)) | ✘ not replaceable · -0.022 [-0.029, -0.016] · 0.79 | ✘ not replaceable · -0.018 [-0.025, -0.012] · 0.79 | ✘ not replaceable · -0.022 [-0.028, -0.016] · 0.79 | ✘ not replaceable · -0.046 [-0.056, -0.037] · 0.74 | ✘ not replaceable · -0.062 [-0.075, -0.051] · 0.71 |

Validation ranking of approaches:

| task   | approach                             |   n_lead_sets |   val_mean_macro_auroc_reduced |
|:-------|:-------------------------------------|--------------:|-------------------------------:|
| t1     | bilstm [S1_scratch]                  |             5 |                       0.886641 |
| t1     | resnet1d [S1_scratch]                |             5 |                       0.886531 |
| t1     | resnet1d-tfft [S2_transfer_finetune] |             5 |                       0.885054 |
| t1     | laresnet [S3_lead_agnostic]          |             5 |                       0.879227 |
| t1     | vit [S1_scratch]                     |             5 |                       0.877944 |
| t1     | resnet1d-tffrz [S2_transfer_frozen]  |             1 |                       0.833465 |

- verdict: {'not_replaceable': 29, 'screen': 1}
- verdict_margin_0.01: {'not_replaceable': 29, 'screen': 1}
- verdict_margin_0.03: {'not_replaceable': 21, 'conditional': 8, 'screen': 1}

## Task T2 — primary approach (chosen on validation): laresnet [S3_lead_agnostic]

| class | limb6 | chest6 | 2lead | leadII | leadI |
|---|---|---|---|---|---|
| ABNORMAL | ✘ not replaceable · -0.025 [-0.032, -0.017] · 0.78 | ✘ not replaceable · -0.021 [-0.027, -0.015] · 0.79 | ✘ not replaceable · -0.024 [-0.032, -0.017] · 0.78 | ✘ not replaceable · -0.053 [-0.062, -0.043] · 0.73 | ✘ not replaceable · -0.065 [-0.078, -0.053] · 0.71 |

Validation ranking of approaches:

| task   | approach                    |   n_lead_sets |   val_mean_macro_auroc_reduced |
|:-------|:----------------------------|--------------:|-------------------------------:|
| t2     | laresnet [S3_lead_agnostic] |             5 |                       0.911283 |
| t2     | resnet1d [S1_scratch]       |             1 |                       0.882823 |

- verdict: {'not_replaceable': 5}
- verdict_margin_0.01: {'not_replaceable': 5}
- verdict_margin_0.03: {'not_replaceable': 4, 'conditional': 1}
