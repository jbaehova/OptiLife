## MIL CV — leave-professor-out

- Protocol: 5-fold GroupKFold (leave-professor-out)
- Baseline HP: `{'hidden_dim': 64, 'dropout': 0.3, 'lr': 0.0005, 'weight_decay': 0.0005, 'max_features': 5000, 'min_df': 2, 'ngram_range': (1, 2)}`
- Best search: `{'hidden_dim': 128, 'dropout': 0.5, 'lr': 0.0005, 'weight_decay': 0.0005, 'max_features': 5000, 'min_df': 2, 'ngram_range': (1, 2)}`
- MAE 개선 (baseline HP → best): **+0.9%**

| Target                   | Model       | MAE           | RMSE          | Pearson r     |
|:-------------------------|:------------|:--------------|:--------------|:--------------|
| workload_label           | Global Mean | 0.587 ± 0.048 | 0.773 ± 0.058 | -             |
| workload_label           | Baseline HP | 0.440 ± 0.018 | 0.577 ± 0.010 | 0.668 ± 0.059 |
| workload_label           | Best Search | 0.439 ± 0.017 | 0.577 ± 0.011 | 0.667 ± 0.059 |
| teamwork_load_label      | Global Mean | 1.027 ± 0.055 | 1.159 ± 0.051 | -             |
| teamwork_load_label      | Baseline HP | 0.671 ± 0.055 | 0.810 ± 0.066 | 0.740 ± 0.036 |
| teamwork_load_label      | Best Search | 0.662 ± 0.057 | 0.804 ± 0.069 | 0.745 ± 0.037 |
| grading_strictness_label | Global Mean | 0.479 ± 0.024 | 0.563 ± 0.022 | -             |
| grading_strictness_label | Baseline HP | 0.417 ± 0.027 | 0.527 ± 0.035 | 0.408 ± 0.124 |
| grading_strictness_label | Best Search | 0.413 ± 0.030 | 0.523 ± 0.037 | 0.424 ± 0.126 |
| average                  | Global Mean | 0.698 ± 0.018 | 0.868 ± 0.022 | 0.346 ± 0.041 |
| average                  | Baseline HP | 0.509 ± 0.018 | 0.651 ± 0.030 | 0.717 ± 0.039 |
| average                  | Best Search | 0.505 ± 0.020 | 0.647 ± 0.031 | 0.720 ± 0.039 |
