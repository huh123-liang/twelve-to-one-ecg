# Key findings for the BS6204 report (T4.5)

These are results on the held-out PTB-XL test fold at 100 Hz. Unless stated otherwise, deep-model summaries use three seeds; the conclusion table averages their predicted probabilities. AUROC differences are absolute changes, not percentages. These are research findings, not a validated clinical screening system.

## F1. Three architectures on 12 leads

- **Number:** T1 test macro AUROC was 0.9227 ± 0.0015 for ResNet1d, 0.9253 ± 0.0013 for CNN-BiLSTM, and 0.9147 ± 0.0009 for ECG-ViT; mean ECE was 0.0333, 0.0392, and 0.0429, respectively. The seed-ensemble ResNet1d minus CNN-BiLSTM difference was +0.0005 (paired-bootstrap 95% CI −0.0020 to +0.0031; Holm-adjusted p = 1.000).
- **Explanation:** All three models discriminated the five diagnostic superclasses well, while the small ResNet1d–CNN-BiLSTM difference does not establish an architecture winner.
- **Research question:** RQ1.
- **Files:** [`table_mean_std.csv`](../results/summary/table_mean_std.csv) (T1, S1, 12lead); [`paired_bootstrap_tests.csv`](../results/summary/paired_bootstrap_tests.csv) (`resnet1d vs bilstm (12lead)`); [`fig1_architecture_12lead.png`](../results/summary/fig1_architecture_12lead.png).

## F2. A stronger traditional baseline still trails the deep model

- **Number:** On T1 12lead, the seed-0 B1 raw-waveform logistic regression scored 0.5116 macro AUROC, the B2 shift-invariant feature baseline scored 0.8629, and the seed-0 ResNet1d scored 0.9211.
- **Explanation:** Hand-designed shift-invariant features rescued much of B1's weak generalization, but a substantial gap to ResNet1d remained; B1 is retained as the pre-specified baseline.
- **Research question:** RQ1.
- **Files:** [`all_runs.csv`](../results/summary/all_runs.csv) (T1, 12lead, B1/B2 and ResNet1d seed 0); [`fig1_architecture_12lead.png`](../results/summary/fig1_architecture_12lead.png).

## F3. Lead reduction has a measurable discrimination cost

- **Number:** Seed-ensemble ResNet1d macro AUROC fell from 0.9298 with 12 leads to 0.8399 with lead I: an absolute drop of 0.0898 (paired-bootstrap 95% CI for the drop, 0.0809–0.0997).
- **Explanation:** A lead-I recording retained some diagnostic signal but lost considerably more information than the full 12-lead input.
- **Research question:** RQ2.
- **Files:** [`seed_ensemble.csv`](../results/summary/seed_ensemble.csv) (T1 ResNet1d, 12lead/leadI); [`paired_bootstrap_tests.csv`](../results/summary/paired_bootstrap_tests.csv) (`resnet1d: 12lead vs leadI`); [`fig2_reduced_leads.png`](../results/summary/fig2_reduced_leads.png).

## F4. Lead I is especially weak for MI in the primary T1 approach

- **Number:** In the validation-selected CNN-BiLSTM approach, MI AUROC fell from 0.9235 (12lead reference) to 0.7959 (leadI), Δ = −0.1275 (paired-bootstrap 95% CI −0.1474 to −0.1074); lead-I sensitivity was 0.4545 at 90% specificity.
- **Explanation:** The pre-specified screen criterion was not met, so this lead-I model cannot replace the 12-lead reference for MI screening in this experiment.
- **Research question:** RQ2.
- **Files:** [`conclusion_long.csv`](../results/summary/conclusion_long.csv) (T1, `bilstm [S1_scratch]`, leadI, MI); [`conclusion_table.md`](../results/summary/conclusion_table.md).

## F5. Chest leads preserve an STTC screening case

- **Number:** Chest6 STTC was marked `screen` for all five evaluated T1 approaches: ΔAUROC ranged from −0.0089 to −0.0045, lower 95% CI bounds from −0.0155 to −0.0097, and sensitivity at 90% specificity from 0.8177 to 0.8369. For the validation-selected CNN-BiLSTM, Δ was −0.0045 (95% CI −0.0097 to +0.0010).
- **Explanation:** This conclusion meets both pre-specified thresholds and is not dependent on which of the five approaches is selected; it applies to chest6, not to a watch-like single lead.
- **Research question:** RQ2.
- **Files:** [`conclusion_long.csv`](../results/summary/conclusion_long.csv) (T1, chest6, STTC, five approaches); [`conclusion_table.md`](../results/summary/conclusion_table.md).

## F6. Transfer fine-tuning did not show a clear advantage

- **Number:** Across the five reduced-lead comparisons with scratch ResNet1d, transfer fine-tuning changed seed-ensemble macro AUROC by −0.0023 to +0.0022; all Holm-adjusted p values were 0.732–1.000. On lead I, LA-ResNet scored 0.0080 lower than scratch ResNet1d (paired-bootstrap 95% CI −0.0124 to −0.0034).
- **Explanation:** These comparisons provide no corrected evidence that fine-tuning beats scratch training; they do not prove the methods equivalent in general. The lead-agnostic model trades a small, sometimes significant cost on reduced leads (−0.0033 to −0.0095 versus five separately trained scratch models; Holm-adjusted p < 0.05 for chest6, leadII and leadI) for deployment convenience: one LA-ResNet checkpoint serves all six lead configurations, and on 12 leads its three-seed mean (0.9264 ± 0.0004) was not lower than ResNet1d (0.9227 ± 0.0015).
- **Research question:** RQ2.
- **Files:** [`paired_bootstrap_tests.csv`](../results/summary/paired_bootstrap_tests.csv) (five `resnet1d-tfft vs resnet1d scratch` rows); [`fig2b_strategies.png`](../results/summary/fig2b_strategies.png).

## F7. Dedicated lead-I triage did not meet the screening target

- **Number:** Lead-I T2 models reached AUROC 0.8777 (ResNet1d) and 0.8754 (LA-ResNet), but sensitivity at 90% specificity was only 0.7055 and 0.7079. The T1-derived triage scores ranged from 0.6998 to 0.7111 on the same sensitivity measure; all lead-I verdicts were `not_replaceable`.
- **Explanation:** Dedicated T2 training gave no apparent lead-I sensitivity gain over deriving abnormality from T1, and neither approach met the 0.80 sensitivity threshold; no formal paired comparison between those T2 and derived scores is claimed here.
- **Research question:** RQ3.
- **Files:** [`conclusion_long.csv`](../results/summary/conclusion_long.csv) (T2 leadI and T1 derived `ABNORMAL` leadI); [`fig10_t2_triage.png`](../results/summary/fig10_t2_triage.png).

## F8. Severe noise changes the model ranking on lead I

- **Number:** At 0 dB test-time noise, three-seed mean T1 lead-I macro AUROC was 0.7585 for ResNet1d versus 0.6864 for CNN-BiLSTM; on clean lead I it was 0.8350 versus 0.8382.
- **Explanation:** CNN-BiLSTM's slight clean-data advantage did not carry over to this severe synthetic-noise condition; this is a descriptive robustness comparison without a noise-specific confidence interval.
- **Research question:** RQ3.
- **Files:** [`all_runs.csv`](../results/summary/all_runs.csv) (T1, S1, leadI, three seeds; `macro_auroc` and `noise_0dB`); [`fig4_noise_robustness.png`](../results/summary/fig4_noise_robustness.png).

## F9. Fewer training labels reduce performance

- **Number:** For the planned seed-0 ResNet1d 12lead runs, training with 10% of the data (1,681 records) gave macro AUROC 0.8814, versus 0.9211 with the full 17,084-record training set; 50% training data gave 0.9120.
- **Number (lead I):** The matching seed-0 ResNet1d lead-I runs gave 0.8119 at 10% and 0.8347 at 100% training data, a gain of only 0.0228, compared with 0.0397 for 12 leads; even 10% of the 12-lead data (0.8814) beat the full-data lead-I model by 0.0467.
- **Explanation:** Label scarcity cost about 0.0397 AUROC at 10% training data on 12 leads. On lead I the curve is much flatter, which suggests that the single-lead gap is limited mainly by the information in the input rather than by the amount of training data. The fraction experiment used seed 0 only, so the exact gaps have no across-seed uncertainty estimate.
- **Research question:** RQ3.
- **Files:** [`all_runs.csv`](../results/summary/all_runs.csv) (T1 ResNet1d, 12lead and leadI, seed 0, fractions 0.10/0.50/1.00); [`fig5_label_efficiency.png`](../results/summary/fig5_label_efficiency.png).

## F10. Model disagreement marks harder records

- **Number:** In the top 20% most-disagreed-on records, mean label error was 0.1958 versus 0.0870 for the remainder on 12lead (2.25×), and 0.2556 versus 0.1410 on leadI (1.81×).
- **Explanation:** Disagreement may help flag records for human review, but this analysis has not tested an actual referral policy or its clinical benefit.
- **Research question:** RQ3.
- **Files:** [`disagreement.csv`](../results/summary/disagreement.csv) (12lead and leadI); [`ensemble_subgroup.md`](../results/summary/ensemble_subgroup.md).

## F11. Masked-signal pretraining helps ECG-ViT with fewer labels

- **Number:** With 10% labeled training data, MSM minus scratch ECG-ViT seed-ensemble AUROC was +0.0065 on 12lead (paired-bootstrap 95% CI +0.0020 to +0.0112) and +0.0149 on leadI (+0.0091 to +0.0210); both comparisons remained significant after Holm correction. Across-seed AUROC standard deviation fell from 0.0078 to 0.0006 on 12lead and from 0.0026 to 0.0005 on leadI. Both pretraining runs reached the 100-epoch limit, with their best validation reconstruction MSE at epochs 98 and 96.
- **Explanation:** MSM narrowed ECG-ViT's low-label gap and reduced seed-to-seed variation, but convergence was not established within the pretraining budget. Its three-seed mean AUROC (0.8759 on 12lead; 0.8096 on leadI) was only comparable to the planned 10%-label ResNet1d seed-0 results (0.8814; 0.8119); this cross-architecture reference is descriptive, not a paired test or evidence that MSM is the best method.
- **Research question:** RQ3.
- **Files:** [`e8_paired_tests.csv`](../results/summary/e8_paired_tests.csv) (ensemble differences); [`e8_self_supervised.csv`](../results/summary/e8_self_supervised.csv) (seed means and standard deviations); [`e8_self_supervised.md`](../results/summary/e8_self_supervised.md) (pretraining epochs); [`all_runs.csv`](../results/summary/all_runs.csv) (10%-label ResNet1d seed 0).

## F12. External triage depends on the diagnosis mapping and operating threshold

- **Number:** Under the v2 main mapping, 34,815 Chapman records (16,092 positive; 18,723 negative) were included. Frozen three-seed T2 ResNet1d AUROC was 0.9552 on 12lead and 0.9159 on leadI, versus 0.9401 and 0.8777 on the PTB-XL test fold. Across v1, v2, and v2_lvhv, external AUROC ranged from 0.9363 to 0.9552 on 12lead and 0.8803 to 0.9159 on leadI. Applying the PTB-XL validation thresholds unchanged yielded external sensitivity/specificity of 0.9054/0.8636 on 12lead and 0.7974/0.9026 on leadI.
- **Explanation:** The frozen model discriminated the restricted v2 normal/abnormal target in this external cohort, but scores changed when left ventricular high voltage was counted as abnormal. The transported threshold did not preserve exactly 90% external specificity, so local calibration would be needed before use. Different case mix and label rules prevent interpreting the higher v2 AUROC as generally better external generalization or revising the PTB-XL lead-I screening verdict.
- **Research question:** RQ3.
- **Files:** [`e7_external_t2.csv`](../results/summary/external_chapman/e7_external_t2.csv) (v2 AUROC, internal comparison, and transported threshold); [`e7_external.md`](../results/summary/external_chapman/e7_external.md) (three-mapping sensitivity table and cohort definition); [`v1 e7_external_t2.csv`](../results/summary/external_chapman_v1_mapping/e7_external_t2.csv); [`v2_lvhv e7_external_t2.csv`](../results/summary/external_chapman_v2_lvhv/e7_external_t2.csv).

## Interpretation limits and literature comparison

- **Explanations:** The ResNet1d receptive field is 1,291 samples, longer than the 1,000-sample input, so its deep Grad-CAM can indicate broad regions but cannot locate an individual ECG wave. The ECG-ViT attention plot has a strong key-token stripe near 4.8–5.0 s that may be an attention sink; do not assign it a physiological mechanism. Sources: [`ResNet1d Grad-CAM`](../results/resnet1d_12lead_frac1_seed0/gradcam.png), [`ViT attention matrix`](../results/vit_12lead_frac1_seed0/attention_matrix.png).
- **Ablations:** Figure 9 contains seed-0 experiments only. For example, the plain RNN scored 0.9279 and the LSTM 0.9258; small single-seed differences do not establish that the RNN is better. Sources: [`all_runs.csv`](../results/summary/all_runs.csv) (T1, 12lead, seed 0, `bilstm-rnn`/`bilstm`) and [`fig9_ablations.png`](../results/summary/fig9_ablations.png).
- **Subgroups:** The lead-I CNN-BiLSTM macro AUROC was 0.8461 in 1,110 male records and 0.8354 in 1,048 female records. These are descriptive estimates without subgroup confidence intervals; they do not establish a fairness difference. Source: [`subgroups.csv`](../results/summary/subgroups.csv).
- **12-lead benchmark:** The published PTB-XL diagnostic-superclass leaderboard lists macro AUROC 0.921–0.930 for several architectures, overlapping our 0.9147–0.9253 three-model range. Architecture, implementation, label handling, and evaluation details differ, so this is context rather than a head-to-head ranking. Sources: [Strodthoff et al. benchmark repository](https://github.com/helme/ecg_ptbxl_benchmarking#4-ptb-xl-diagnostic-superclasses) and [`table_mean_std.csv`](../results/summary/table_mean_std.csv).
- **Feature baseline in the benchmark:** The same leaderboard lists a Wavelet+NN feature-based model at 0.874, close to our B2 shift-invariant feature baseline (0.8629), which supports B2 as a reasonable traditional reference. Source: [Strodthoff et al. benchmark repository](https://github.com/helme/ecg_ptbxl_benchmarking#4-ptb-xl-diagnostic-superclasses).
- **Reporting p values:** Bootstrap p values use 1,000 resamples, so a stored value of 0 means p < 0.001; write "p < 0.001", never "p = 0". Report Holm-adjusted values (`p_holm`) when several comparisons are discussed together.
- **Lead-I MI literature:** A 2024 PTB-XL lead-I acute-MI binary-classification study reported test AUC 0.92. Our five-superclass multi-label MI value is 0.7959 for the validation-selected lead-I approach; the target and cohort construction differ, so the values should not be read as a direct replication or superiority test. Sources: [Davarmanesh et al. preprint](https://www.medrxiv.org/content/10.1101/2024.10.15.24315544v1.full.pdf) and [`conclusion_long.csv`](../results/summary/conclusion_long.csv).
