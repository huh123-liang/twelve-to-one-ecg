# table5 self supervised

Sources: results/summary/e8_self_supervised.csv and e8_paired_tests.csv. Each row is one lead configuration; mean ± SD describes three seeds, followed by seed-ensemble AUROC [95% CI]. Paired difference is MSM minus scratch on seed-ensemble predictions with 1,000 record bootstraps. Full-data ViT is a reference, not part of the paired test.

| leads   | scratch (mean ± SD; ensemble [95% CI])   | MSM (mean ± SD; ensemble [95% CI])       | MSM - scratch [95% CI]     | p       |   full-data ViT reference |
|:--------|:-----------------------------------------|:-----------------------------------------|:---------------------------|:--------|--------------------------:|
| 12lead  | 0.8668 ± 0.0078; 0.8860 [0.8771, 0.8955] | 0.8759 ± 0.0006; 0.8924 [0.8837, 0.9012] | +0.0065 [+0.0020, +0.0112] | 0.006   |                    0.9221 |
| leadI   | 0.7919 ± 0.0026; 0.8053 [0.7938, 0.8185] | 0.8096 ± 0.0005; 0.8203 [0.8087, 0.8318] | +0.0149 [+0.0091, +0.0210] | < 0.001 |                    0.8287 |
