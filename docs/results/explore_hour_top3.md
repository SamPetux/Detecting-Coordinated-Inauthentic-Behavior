# Exploration summary: `hour_top3`

Runs: 13 files. Mean over seeds (generator A); generator C (unseen) shown separately.

## Generator A: recall@0.001

| detector        |    L0 |    L1 |    L2 |    L3 |    L4 |    L5 |
|:----------------|------:|------:|------:|------:|------:|------:|
| full            | 1     | 1     | 1     | 0.99  | 0.55  | 0.593 |
| full_noM        | 1     | 1     | 1     | 0.986 | 0.545 | 0.571 |
| full_noMY       | 1     | 1     | 0.995 | 0.955 | 0.421 | 0.524 |
| layer_T         | 0.788 | 0.781 | 0.776 | 0.776 | 0.188 | 0.224 |
| layer_S         | 1     | 1     | 0.998 | 0.967 | 0.526 | 0.588 |
| layer_L         | 1     | 1     | 0     | 0.019 | 0     | 0.007 |
| raw_T           | 0.381 | 0.381 | 0.381 | 0.381 | 0.055 | 0     |
| raw_S           | 1     | 1     | 0.995 | 0.933 | 0.498 | 0.574 |
| copypaste       | 1     | 1     | 0     | 0.002 | 0     | 0.01  |
| account_machine | 0.681 | 0.762 | 0.714 | 0.514 | 0.438 | 0.371 |

## Generator A: recall@0.01

| detector        |    L0 |    L1 |    L2 |    L3 |    L4 |    L5 |
|:----------------|------:|------:|------:|------:|------:|------:|
| full            | 1     | 1     | 1     | 0.998 | 0.631 | 0.721 |
| full_noM        | 1     | 1     | 1     | 0.998 | 0.629 | 0.714 |
| full_noMY       | 1     | 1     | 1     | 0.995 | 0.564 | 0.686 |
| layer_T         | 0.993 | 0.993 | 0.993 | 0.99  | 0.438 | 0.548 |
| layer_S         | 1     | 1     | 0.998 | 0.967 | 0.529 | 0.593 |
| layer_L         | 1     | 1     | 0     | 0.019 | 0     | 0.007 |
| raw_T           | 0.795 | 0.795 | 0.795 | 0.795 | 0.26  | 0.231 |
| raw_S           | 1     | 1     | 1     | 0.99  | 0.69  | 0.793 |
| copypaste       | 1     | 1     | 0.019 | 0.017 | 0     | 0.014 |
| account_machine | 1     | 1     | 1     | 1     | 1     | 1     |

## Generator A: auprc

| detector        |    L0 |    L1 |    L2 |    L3 |    L4 |    L5 |
|:----------------|------:|------:|------:|------:|------:|------:|
| full            | 0.987 | 0.987 | 0.947 | 0.922 | 0.51  | 0.587 |
| full_noM        | 0.986 | 0.986 | 0.946 | 0.919 | 0.503 | 0.576 |
| full_noMY       | 0.941 | 0.941 | 0.929 | 0.912 | 0.423 | 0.546 |
| layer_T         | 0.828 | 0.825 | 0.826 | 0.829 | 0.247 | 0.314 |
| layer_S         | 0.937 | 0.935 | 0.932 | 0.895 | 0.45  | 0.526 |
| layer_L         | 1     | 1     | 0.007 | 0.022 | 0.007 | 0.013 |
| raw_T           | 0.557 | 0.557 | 0.556 | 0.555 | 0.115 | 0.105 |
| raw_S           | 0.993 | 0.993 | 0.985 | 0.935 | 0.538 | 0.624 |
| copypaste       | 1     | 1     | 0.009 | 0.009 | 0.007 | 0.013 |
| account_machine | 0.723 | 0.752 | 0.723 | 0.663 | 0.625 | 0.578 |

## Generator A: benign_flag@0.01

| detector        |    L0 |    L1 |    L2 |    L3 |    L4 |    L5 |
|:----------------|------:|------:|------:|------:|------:|------:|
| full            | 0     | 0     | 0     | 0     | 0     | 0.004 |
| full_noM        | 0     | 0     | 0     | 0     | 0     | 0     |
| full_noMY       | 0.002 | 0.002 | 0.002 | 0     | 0.004 | 0.002 |
| layer_T         | 0.002 | 0.002 | 0     | 0.006 | 0.004 | 0.002 |
| layer_S         | 0     | 0     | 0.002 | 0     | 0     | 0     |
| layer_L         | 0     | 0     | 0     | 0     | 0     | 0     |
| raw_T           | 0     | 0     | 0     | 0     | 0     | 0     |
| raw_S           | 0.025 | 0.025 | 0.027 | 0.029 | 0.033 | 0.037 |
| copypaste       | 0     | 0     | 0     | 0     | 0     | 0     |
| account_machine | 0.223 | 0.223 | 0.223 | 0.223 | 0.223 | 0.223 |

## Generator A: h3_ratio@0.01

| detector        |      L0 |      L1 |      L2 |      L3 |      L4 |      L5 |
|:----------------|--------:|--------:|--------:|--------:|--------:|--------:|
| full            |   0.284 |   0.284 |   0.284 |   0.284 |   0.266 |   0.278 |
| full_noM        | nan     | nan     | nan     | nan     | nan     | nan     |
| full_noMY       |   0.274 |   0.297 |   0.263 |   0.253 |   0.246 |   0.248 |
| layer_T         |   0.297 |   0.304 |   0.274 |   0.274 |   0.286 |   0.248 |
| layer_S         |   0.784 |   0.784 |   0.784 |   0.784 |   0.851 |   0.767 |
| layer_L         | nan     | nan     | nan     | nan     | nan     | nan     |
| raw_T           |   1.187 |   1.187 |   1.187 |   1.188 |   1.39  |   1.404 |
| raw_S           |   0.748 |   0.751 |   0.745 |   0.759 |   0.758 |   0.706 |
| copypaste       | nan     | nan     | nan     | nan     | nan     | nan     |
| account_machine | nan     | nan     | nan     | nan     | nan     | nan     |

## Generator C: recall@0.001

| detector        |    L3 |
|:----------------|------:|
| full            | 1     |
| full_noM        | 1     |
| full_noMY       | 0.967 |
| layer_T         | 0.762 |
| layer_S         | 0.99  |
| layer_L         | 0     |
| raw_T           | 0.481 |
| raw_S           | 0.971 |
| copypaste       | 0.1   |
| account_machine | 0.062 |

## Generator C: recall@0.01

| detector        |    L3 |
|:----------------|------:|
| full            | 1     |
| full_noM        | 1     |
| full_noMY       | 1     |
| layer_T         | 1     |
| layer_S         | 0.99  |
| layer_L         | 0     |
| raw_T           | 0.79  |
| raw_S           | 1     |
| copypaste       | 0.276 |
| account_machine | 1     |

## Generator C: auprc

| detector        |    L3 |
|:----------------|------:|
| full            | 0.939 |
| full_noM        | 0.938 |
| full_noMY       | 0.924 |
| layer_T         | 0.854 |
| layer_S         | 0.923 |
| layer_L         | 0.007 |
| raw_T           | 0.558 |
| raw_S           | 0.973 |
| copypaste       | 0.198 |
| account_machine | 0.449 |

## Generator C: benign_flag@0.01

| detector        |    L3 |
|:----------------|------:|
| full            | 0.018 |
| full_noM        | 0.018 |
| full_noMY       | 0.009 |
| layer_T         | 0.009 |
| layer_S         | 0     |
| layer_L         | 0     |
| raw_T           | 0     |
| raw_S           | 0.045 |
| copypaste       | 0     |
| account_machine | 0.936 |

## Generator C: h3_ratio@0.01

| detector        |      L3 |
|:----------------|--------:|
| full            |   0.27  |
| full_noM        | nan     |
| full_noMY       |   0.291 |
| layer_T         |   0.298 |
| layer_S         |   0.775 |
| layer_L         | nan     |
| raw_T           |   1.139 |
| raw_S           |   0.729 |
| copypaste       | nan     |
| account_machine | nan     |

