## MIL CV — leave-course-out

- Protocol: 5-fold GroupKFold (leave-course-out)
- Baseline HP: `{'hidden_dim': 64, 'dropout': 0.3, 'lr': 0.0005, 'weight_decay': 0.0005, 'max_features': 5000, 'min_df': 2, 'ngram_range': (1, 2)}`
- Best search: `{'hidden_dim': 64, 'dropout': 0.5, 'lr': 0.001, 'weight_decay': 0.0005, 'max_features': 5000, 'min_df': 1, 'ngram_range': (1, 2)}`
- MAE 개선 (baseline HP → best): **+0.8%**

| Target                   | Model       | MAE           | RMSE          | Pearson r     |
|:-------------------------|:------------|:--------------|:--------------|:--------------|
| workload_label           | Global Mean | 0.585 ± 0.034 | 0.777 ± 0.052 | -             |
| workload_label           | Baseline HP | 0.437 ± 0.058 | 0.570 ± 0.068 | 0.686 ± 0.050 |
| workload_label           | Best Search | 0.436 ± 0.064 | 0.572 ± 0.076 | 0.684 ± 0.057 |
| teamwork_load_label      | Global Mean | 1.035 ± 0.075 | 1.159 ± 0.074 | -             |
| teamwork_load_label      | Baseline HP | 0.667 ± 0.030 | 0.810 ± 0.041 | 0.735 ± 0.056 |
| teamwork_load_label      | Best Search | 0.657 ± 0.028 | 0.806 ± 0.040 | 0.736 ± 0.055 |
| grading_strictness_label | Global Mean | 0.479 ± 0.050 | 0.560 ± 0.054 | -             |
| grading_strictness_label | Baseline HP | 0.405 ± 0.042 | 0.518 ± 0.031 | 0.442 ± 0.062 |
| grading_strictness_label | Best Search | 0.404 ± 0.043 | 0.516 ± 0.033 | 0.457 ± 0.066 |
| average                  | Global Mean | 0.699 ± 0.015 | 0.870 ± 0.021 | 0.345 ± 0.037 |
| average                  | Baseline HP | 0.503 ± 0.023 | 0.647 ± 0.017 | 0.718 ± 0.037 |
| average                  | Best Search | 0.499 ± 0.025 | 0.646 ± 0.018 | 0.721 ± 0.038 |
