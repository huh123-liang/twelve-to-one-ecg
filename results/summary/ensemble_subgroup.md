# Ensembles and subgroups

## Seed ensemble

| task   | variant        | strategy             | leads   |   n_seeds |   single_auroc_mean |   ensemble_auroc |   single_ece_mean |   ensemble_ece |
|:-------|:---------------|:---------------------|:--------|----------:|--------------------:|-----------------:|------------------:|---------------:|
| t1     | bilstm         | S1_scratch           | 12lead  |         3 |              0.9253 |           0.9292 |            0.0392 |         0.0309 |
| t1     | bilstm         | S1_scratch           | 2lead   |         3 |              0.8937 |           0.8971 |            0.0312 |         0.0234 |
| t1     | bilstm         | S1_scratch           | chest6  |         3 |              0.9018 |           0.9053 |            0.0387 |         0.0314 |
| t1     | bilstm         | S1_scratch           | leadI   |         3 |              0.8382 |           0.841  |            0.0354 |         0.0296 |
| t1     | bilstm         | S1_scratch           | leadII  |         3 |              0.8544 |           0.8575 |            0.0354 |         0.0317 |
| t1     | bilstm         | S1_scratch           | limb6   |         3 |              0.8929 |           0.8966 |            0.0421 |         0.0364 |
| t1     | laresnet       | S3_lead_agnostic     | 12lead  |         3 |              0.9264 |           0.9313 |            0.0399 |         0.0307 |
| t1     | laresnet       | S3_lead_agnostic     | 2lead   |         3 |              0.8895 |           0.8933 |            0.028  |         0.0254 |
| t1     | laresnet       | S3_lead_agnostic     | chest6  |         3 |              0.8944 |           0.9    |            0.0313 |         0.0264 |
| t1     | laresnet       | S3_lead_agnostic     | leadI   |         3 |              0.8291 |           0.8319 |            0.0311 |         0.0244 |
| t1     | laresnet       | S3_lead_agnostic     | leadII  |         3 |              0.8448 |           0.8491 |            0.0278 |         0.0253 |
| t1     | laresnet       | S3_lead_agnostic     | limb6   |         3 |              0.8887 |           0.8942 |            0.0313 |         0.0228 |
| t1     | resnet1d       | S1_scratch           | 12lead  |         3 |              0.9227 |           0.9298 |            0.0333 |         0.0276 |
| t1     | resnet1d       | S1_scratch           | 2lead   |         3 |              0.8902 |           0.8967 |            0.0445 |         0.0299 |
| t1     | resnet1d       | S1_scratch           | chest6  |         3 |              0.8982 |           0.9054 |            0.0471 |         0.036  |
| t1     | resnet1d       | S1_scratch           | leadI   |         3 |              0.835  |           0.8399 |            0.0459 |         0.0375 |
| t1     | resnet1d       | S1_scratch           | leadII  |         3 |              0.8523 |           0.8587 |            0.0329 |         0.0242 |
| t1     | resnet1d       | S1_scratch           | limb6   |         3 |              0.8933 |           0.8986 |            0.0361 |         0.0282 |
| t1     | resnet1d-tffrz | S2_transfer_frozen   | leadI   |         3 |              0.8288 |           0.8321 |            0.0344 |         0.031  |
| t1     | resnet1d-tfft  | S2_transfer_finetune | 2lead   |         3 |              0.8929 |           0.8978 |            0.038  |         0.0316 |
| t1     | resnet1d-tfft  | S2_transfer_finetune | chest6  |         3 |              0.9023 |           0.9076 |            0.0383 |         0.0285 |
| t1     | resnet1d-tfft  | S2_transfer_finetune | leadI   |         3 |              0.8336 |           0.8376 |            0.0413 |         0.0375 |
| t1     | resnet1d-tfft  | S2_transfer_finetune | leadII  |         3 |              0.8524 |           0.8569 |            0.0425 |         0.0379 |
| t1     | resnet1d-tfft  | S2_transfer_finetune | limb6   |         3 |              0.8923 |           0.8983 |            0.0331 |         0.0289 |
| t1     | vit            | S1_scratch           | 12lead  |         3 |              0.9147 |           0.9221 |            0.0429 |         0.0325 |
| t1     | vit            | S1_scratch           | 2lead   |         3 |              0.8844 |           0.8903 |            0.0371 |         0.0333 |
| t1     | vit            | S1_scratch           | chest6  |         3 |              0.891  |           0.8975 |            0.0358 |         0.0294 |
| t1     | vit            | S1_scratch           | leadI   |         3 |              0.8238 |           0.8287 |            0.0366 |         0.0298 |
| t1     | vit            | S1_scratch           | leadII  |         3 |              0.8409 |           0.8458 |            0.046  |         0.0419 |
| t1     | vit            | S1_scratch           | limb6   |         3 |              0.8833 |           0.8896 |            0.0435 |         0.0385 |
| t2     | laresnet       | S3_lead_agnostic     | 12lead  |         3 |              0.9386 |           0.9408 |            0.0493 |         0.0459 |
| t2     | laresnet       | S3_lead_agnostic     | 2lead   |         3 |              0.9139 |           0.9159 |            0.0546 |         0.0546 |
| t2     | laresnet       | S3_lead_agnostic     | chest6  |         3 |              0.9158 |           0.9187 |            0.0417 |         0.0387 |
| t2     | laresnet       | S3_lead_agnostic     | leadI   |         3 |              0.8732 |           0.8754 |            0.0572 |         0.056  |
| t2     | laresnet       | S3_lead_agnostic     | leadII  |         3 |              0.8844 |           0.8874 |            0.0364 |         0.0297 |
| t2     | laresnet       | S3_lead_agnostic     | limb6   |         3 |              0.9129 |           0.9156 |            0.0424 |         0.0411 |
| t2     | resnet1d       | S1_scratch           | 12lead  |         3 |              0.9364 |           0.9401 |            0.0519 |         0.0355 |
| t2     | resnet1d       | S1_scratch           | leadI   |         3 |              0.8748 |           0.8777 |            0.0267 |         0.0255 |

## Heterogeneous ensemble (ResNet1d + CNN-BiLSTM + ECG-ViT)

| leads   |   resnet1d_auroc |   bilstm_auroc |   vit_auroc |   hetero_ensemble_auroc |   hetero_ensemble_ece |
|:--------|-----------------:|---------------:|------------:|------------------------:|----------------------:|
| 12lead  |           0.9298 |         0.9292 |      0.9221 |                  0.9315 |                0.0268 |
| limb6   |           0.8986 |         0.8966 |      0.8896 |                  0.9005 |                0.0299 |
| chest6  |           0.9054 |         0.9053 |      0.8975 |                  0.9078 |                0.0278 |
| 2lead   |           0.8967 |         0.8971 |      0.8903 |                  0.8992 |                0.0234 |
| leadII  |           0.8587 |         0.8575 |      0.8458 |                  0.8594 |                0.028  |
| leadI   |           0.8399 |         0.841  |      0.8287 |                  0.8416 |                0.0283 |

Records where the architectures disagree most (top 20%) vs the rest:

| leads   |   error_rate_top20pct_disagreement |   error_rate_rest |   auroc_top20pct_disagreement |   auroc_rest |
|:--------|-----------------------------------:|------------------:|------------------------------:|-------------:|
| 12lead  |                             0.1958 |            0.087  |                        0.8638 |       0.9364 |
| limb6   |                             0.2176 |            0.1104 |                        0.8335 |       0.905  |
| chest6  |                             0.2051 |            0.1012 |                        0.8437 |       0.911  |
| 2lead   |                             0.2176 |            0.1111 |                        0.8355 |       0.9032 |
| leadII  |                             0.237  |            0.1328 |                        0.7932 |       0.8569 |
| leadI   |                             0.2556 |            0.141  |                        0.7676 |       0.8391 |

## Subgroups (seed-ensembled predictions)

| approach                    | leads   | subgroup   |    n |   macro_auroc |   auroc_NORM |   auroc_MI |   auroc_STTC |   auroc_CD |   auroc_HYP |
|:----------------------------|:--------|:-----------|-----:|--------------:|-------------:|-----------:|-------------:|-----------:|------------:|
| resnet1d [S1_scratch]       | 12lead  | sex=male   | 1110 |        0.9383 |       0.957  |     0.9382 |       0.9511 |     0.9313 |      0.9138 |
| resnet1d [S1_scratch]       | 12lead  | sex=female | 1048 |        0.9199 |       0.9424 |     0.9113 |       0.9244 |     0.9122 |      0.9093 |
| resnet1d [S1_scratch]       | 12lead  | age <50    |  521 |        0.927  |       0.9097 |     0.9654 |       0.9563 |     0.874  |      0.9298 |
| resnet1d [S1_scratch]       | 12lead  | age 50-70  |  924 |        0.9245 |       0.9332 |     0.9118 |       0.9458 |     0.9185 |      0.9132 |
| resnet1d [S1_scratch]       | 12lead  | age >70    |  713 |        0.9137 |       0.9596 |     0.892  |       0.8974 |     0.9287 |      0.8906 |
| resnet1d [S1_scratch]       | leadI   | sex=male   | 1110 |        0.8467 |       0.8814 |     0.8065 |       0.9018 |     0.8333 |      0.8107 |
| resnet1d [S1_scratch]       | leadI   | sex=female | 1048 |        0.833  |       0.8959 |     0.7738 |       0.8548 |     0.8212 |      0.8193 |
| resnet1d [S1_scratch]       | leadI   | age <50    |  521 |        0.818  |       0.8188 |     0.8333 |       0.8765 |     0.7563 |      0.8051 |
| resnet1d [S1_scratch]       | leadI   | age 50-70  |  924 |        0.8155 |       0.8336 |     0.7345 |       0.8858 |     0.8083 |      0.8154 |
| resnet1d [S1_scratch]       | leadI   | age >70    |  713 |        0.8147 |       0.9082 |     0.728  |       0.8187 |     0.8435 |      0.7751 |
| laresnet [S3_lead_agnostic] | 12lead  | sex=male   | 1110 |        0.9382 |       0.9561 |     0.9376 |       0.9489 |     0.933  |      0.9151 |
| laresnet [S3_lead_agnostic] | 12lead  | sex=female | 1048 |        0.9232 |       0.9457 |     0.9156 |       0.9282 |     0.9159 |      0.9108 |
| laresnet [S3_lead_agnostic] | 12lead  | age <50    |  521 |        0.9236 |       0.9058 |     0.9626 |       0.9539 |     0.8704 |      0.9254 |
| laresnet [S3_lead_agnostic] | 12lead  | age 50-70  |  924 |        0.9271 |       0.9338 |     0.9124 |       0.9452 |     0.9225 |      0.9216 |
| laresnet [S3_lead_agnostic] | 12lead  | age >70    |  713 |        0.9166 |       0.9627 |     0.8955 |       0.9044 |     0.9341 |      0.8865 |
| laresnet [S3_lead_agnostic] | leadI   | sex=male   | 1110 |        0.8387 |       0.8836 |     0.8009 |       0.8982 |     0.8018 |      0.809  |
| laresnet [S3_lead_agnostic] | leadI   | sex=female | 1048 |        0.8248 |       0.8944 |     0.7618 |       0.8489 |     0.7988 |      0.8198 |
| laresnet [S3_lead_agnostic] | leadI   | age <50    |  521 |        0.8049 |       0.8171 |     0.7946 |       0.8797 |     0.7137 |      0.8192 |
| laresnet [S3_lead_agnostic] | leadI   | age 50-70  |  924 |        0.805  |       0.8385 |     0.7268 |       0.8837 |     0.7574 |      0.8187 |
| laresnet [S3_lead_agnostic] | leadI   | age >70    |  713 |        0.8061 |       0.9017 |     0.7174 |       0.8109 |     0.8355 |      0.7648 |
| vit [S1_scratch]            | 12lead  | sex=male   | 1110 |        0.9316 |       0.9526 |     0.9348 |       0.9492 |     0.9208 |      0.9008 |
| vit [S1_scratch]            | 12lead  | sex=female | 1048 |        0.9113 |       0.9376 |     0.905  |       0.9217 |     0.8967 |      0.8957 |
| vit [S1_scratch]            | 12lead  | age <50    |  521 |        0.9217 |       0.9005 |     0.9511 |       0.9666 |     0.8631 |      0.927  |
| vit [S1_scratch]            | 12lead  | age 50-70  |  924 |        0.9165 |       0.926  |     0.906  |       0.9428 |     0.8993 |      0.9082 |
| vit [S1_scratch]            | 12lead  | age >70    |  713 |        0.9024 |       0.9548 |     0.8881 |       0.8886 |     0.9176 |      0.8628 |
| vit [S1_scratch]            | leadI   | sex=male   | 1110 |        0.8341 |       0.8837 |     0.7935 |       0.9005 |     0.7936 |      0.799  |
| vit [S1_scratch]            | leadI   | sex=female | 1048 |        0.8233 |       0.8921 |     0.7678 |       0.8455 |     0.8044 |      0.8067 |
| vit [S1_scratch]            | leadI   | age <50    |  521 |        0.8026 |       0.826  |     0.8228 |       0.8772 |     0.7009 |      0.7862 |
| vit [S1_scratch]            | leadI   | age 50-70  |  924 |        0.8016 |       0.8315 |     0.7208 |       0.8838 |     0.7652 |      0.8068 |
| vit [S1_scratch]            | leadI   | age >70    |  713 |        0.8004 |       0.9016 |     0.7095 |       0.8044 |     0.8299 |      0.7567 |
| bilstm [S1_scratch]         | 12lead  | sex=male   | 1110 |        0.9387 |       0.9543 |     0.9357 |       0.9533 |     0.9388 |      0.9117 |
| bilstm [S1_scratch]         | 12lead  | sex=female | 1048 |        0.9187 |       0.9408 |     0.908  |       0.9215 |     0.9161 |      0.9071 |
| bilstm [S1_scratch]         | 12lead  | age <50    |  521 |        0.929  |       0.9053 |     0.9645 |       0.9606 |     0.8852 |      0.9296 |
| bilstm [S1_scratch]         | 12lead  | age 50-70  |  924 |        0.9242 |       0.93   |     0.9102 |       0.9414 |     0.9245 |      0.9151 |
| bilstm [S1_scratch]         | 12lead  | age >70    |  713 |        0.9098 |       0.9528 |     0.8838 |       0.9017 |     0.9338 |      0.8769 |
| bilstm [S1_scratch]         | leadI   | sex=male   | 1110 |        0.8461 |       0.8846 |     0.8076 |       0.9069 |     0.8228 |      0.8083 |
| bilstm [S1_scratch]         | leadI   | sex=female | 1048 |        0.8354 |       0.8951 |     0.7816 |       0.8517 |     0.8233 |      0.825  |
| bilstm [S1_scratch]         | leadI   | age <50    |  521 |        0.8105 |       0.8221 |     0.8108 |       0.8796 |     0.7314 |      0.8085 |
| bilstm [S1_scratch]         | leadI   | age 50-70  |  924 |        0.8147 |       0.8365 |     0.7364 |       0.8906 |     0.7897 |      0.8204 |
| bilstm [S1_scratch]         | leadI   | age >70    |  713 |        0.8194 |       0.9054 |     0.7438 |       0.8193 |     0.8578 |      0.7707 |
