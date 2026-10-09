# Table 3: strategy paired tests

Source: strategy rows of results/summary/paired_bootstrap_tests.csv. Differences are paired seed-ensemble AUROC comparisons. Holm adjustment covers all 29 comparisons in the source file, although only strategy-related rows are shown here. Values stored as zero after 1,000 bootstraps are displayed as < 0.001.

| comparison                                |    diff | p_value   |   ci95_lo |   ci95_hi | p_holm   |
|:------------------------------------------|--------:|:----------|----------:|----------:|:---------|
| limb6: resnet1d-tfft vs resnet1d scratch  | -0.0003 | 0.788     |   -0.0031 |    0.0025 | 1.000    |
| limb6: laresnet vs resnet1d scratch       | -0.0044 | 0.008     |   -0.0079 |   -0.0011 | 0.064    |
| chest6: resnet1d-tfft vs resnet1d scratch |  0.0022 | 0.146     |   -0.0007 |    0.005  | 0.732    |
| chest6: laresnet vs resnet1d scratch      | -0.0054 | 0.002     |   -0.0092 |   -0.0018 | 0.018    |
| 2lead: resnet1d-tfft vs resnet1d scratch  |  0.0011 | 0.462     |   -0.0017 |    0.0041 | 1.000    |
| 2lead: laresnet vs resnet1d scratch       | -0.0033 | 0.066     |   -0.007  |    0.0002 | 0.462    |
| leadII: resnet1d-tfft vs resnet1d scratch | -0.0017 | 0.316     |   -0.0052 |    0.0018 | 1.000    |
| leadII: laresnet vs resnet1d scratch      | -0.0095 | < 0.001   |   -0.0148 |   -0.0045 | < 0.001  |
| leadI: resnet1d-tfft vs resnet1d scratch  | -0.0023 | 0.122     |   -0.0052 |    0.0006 | 0.732    |
| leadI: resnet1d-tffrz vs resnet1d scratch | -0.0078 | < 0.001   |   -0.0117 |   -0.0037 | < 0.001  |
| leadI: laresnet vs resnet1d scratch       | -0.008  | < 0.001   |   -0.0124 |   -0.0034 | < 0.001  |
