# Table 2: 12-lead models

Sources: results/summary/table_mean_std.csv, seed_ensemble.csv, and aligned three-seed test predictions in results/<run>/preds.npz. Bootstrap CI: 1,000 record resamples, seed 0. Single-seed baselines have no across-seed SD or ensemble CI.

| Model       |   Seeds | AUROC           | AUPRC           | Sens@spec90     | ECE             | Ensemble AUROC [95% CI]   |
|:------------|--------:|:----------------|:----------------|:----------------|:----------------|:--------------------------|
| B0-PRIOR    |       1 | 0.5000 (1 seed) | 0.2588 (1 seed) | 0.0000 (1 seed) | 0.0021 (1 seed) | —                         |
| B1-LOGREG   |       1 | 0.5116 (1 seed) | 0.2865 (1 seed) | 0.1424 (1 seed) | 0.1107 (1 seed) | —                         |
| B2 features |       1 | 0.8629 (1 seed) | 0.7005 (1 seed) | 0.6268 (1 seed) | 0.0278 (1 seed) | —                         |
| ResNet1d    |       3 | 0.9227 ± 0.0015 | 0.8160 ± 0.0039 | 0.7768 ± 0.0102 | 0.0333 ± 0.0009 | 0.9298 [0.9231, 0.9367]   |
| CNN-BiLSTM  |       3 | 0.9253 ± 0.0013 | 0.8174 ± 0.0026 | 0.7730 ± 0.0029 | 0.0392 ± 0.0058 | 0.9292 [0.9225, 0.9363]   |
| ECG-ViT     |       3 | 0.9147 ± 0.0009 | 0.7953 ± 0.0013 | 0.7517 ± 0.0070 | 0.0429 ± 0.0046 | 0.9221 [0.9152, 0.9295]   |
