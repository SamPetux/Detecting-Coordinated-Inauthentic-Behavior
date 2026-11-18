# Exploration summary: `day_max`

Runs: 13 files. Mean over seeds (generator A); generator C (unseen) shown separately.

## Generator A: recall@0.001

| detector        |    L0 |    L1 |    L2 |    L3 |    L4 |    L5 |
|:----------------|------:|------:|------:|------:|------:|------:|
| full            | 1     | 1     | 1     | 0.988 | 0.598 | 0.624 |
| full_noM        | 1     | 1     | 0.998 | 0.988 | 0.588 | 0.595 |
| full_noMY       | 1     | 1     | 0.998 | 0.976 | 0.44  | 0.495 |
| layer_T         | 0.96  | 0.957 | 0.964 | 0.952 | 0.329 | 0.26  |
| layer_S         | 1     | 1     | 0.998 | 0.967 | 0.529 | 0.593 |
| layer_L         | 1     | 1     | 0     | 0.019 | 0     | 0.007 |
| raw_T           | 0.381 | 0.381 | 0.381 | 0.381 | 0.055 | 0     |
| raw_S           | 1     | 1     | 0.995 | 0.933 | 0.498 | 0.574 |
| copypaste       | 1     | 1     | 0     | 0.002 | 0     | 0.01  |
| account_machine | 0.681 | 0.762 | 0.714 | 0.514 | 0.438 | 0.371 |

## Generator A: recall@0.01

| detector        |    L0 |    L1 |    L2 |    L3 |    L4 |    L5 |
|:----------------|------:|------:|------:|------:|------:|------:|
| full            | 1     | 1     | 1     | 1     | 0.633 | 0.733 |
| full_noM        | 1     | 1     | 1     | 1     | 0.633 | 0.731 |
| full_noMY       | 1     | 1     | 1     | 1     | 0.607 | 0.714 |
| layer_T         | 0.995 | 0.995 | 0.995 | 0.995 | 0.486 | 0.538 |
| layer_S         | 1     | 1     | 0.998 | 0.967 | 0.529 | 0.593 |
| layer_L         | 1     | 1     | 0     | 0.019 | 0     | 0.007 |
| raw_T           | 0.795 | 0.795 | 0.795 | 0.795 | 0.26  | 0.231 |
| raw_S           | 1     | 1     | 1     | 0.99  | 0.69  | 0.793 |
| copypaste       | 1     | 1     | 0.019 | 0.017 | 0     | 0.014 |
| account_machine | 1     | 1     | 1     | 1     | 1     | 1     |

## Generator A: auprc

| detector        |    L0 |    L1 |    L2 |    L3 |    L4 |    L5 |
|:----------------|------:|------:|------:|------:|------:|------:|
| full            | 0.924 | 0.926 | 0.909 | 0.899 | 0.512 | 0.584 |
| full_noM        | 0.924 | 0.926 | 0.909 | 0.897 | 0.507 | 0.574 |
| full_noMY       | 0.943 | 0.943 | 0.934 | 0.921 | 0.45  | 0.528 |
| layer_T         | 0.948 | 0.952 | 0.947 | 0.943 | 0.359 | 0.34  |
| layer_S         | 0.94  | 0.939 | 0.93  | 0.888 | 0.436 | 0.502 |
| layer_L         | 1     | 1     | 0.007 | 0.026 | 0.007 | 0.014 |
| raw_T           | 0.557 | 0.557 | 0.556 | 0.555 | 0.115 | 0.105 |
| raw_S           | 0.993 | 0.993 | 0.985 | 0.935 | 0.538 | 0.624 |
| copypaste       | 1     | 1     | 0.009 | 0.009 | 0.007 | 0.013 |
| account_machine | 0.723 | 0.752 | 0.723 | 0.663 | 0.625 | 0.578 |

## Generator A: benign_flag@0.01

| detector        |    L0 |    L1 |    L2 |    L3 |    L4 |    L5 |
|:----------------|------:|------:|------:|------:|------:|------:|
| full            | 0.012 | 0.01  | 0.012 | 0.01  | 0.008 | 0.01  |
| full_noM        | 0.012 | 0.01  | 0.012 | 0.01  | 0.008 | 0.01  |
| full_noMY       | 0.019 | 0.019 | 0.019 | 0.021 | 0.027 | 0.019 |
| layer_T         | 0.023 | 0.019 | 0.017 | 0.021 | 0.027 | 0.021 |
| layer_S         | 0     | 0     | 0.002 | 0     | 0     | 0     |
| layer_L         | 0     | 0     | 0     | 0     | 0     | 0     |
| raw_T           | 0     | 0     | 0     | 0     | 0     | 0     |
| raw_S           | 0.025 | 0.025 | 0.027 | 0.029 | 0.033 | 0.037 |
| copypaste       | 0     | 0     | 0     | 0     | 0     | 0     |
| account_machine | 0.223 | 0.223 | 0.223 | 0.223 | 0.223 | 0.223 |

## Generator A: h3_ratio@0.01

| detector        |      L0 |      L1 |      L2 |      L3 |      L4 |      L5 |
|:----------------|--------:|--------:|--------:|--------:|--------:|--------:|
| full            |   0.378 |   0.367 |   0.367 |   0.346 |   0.346 |   0.359 |
| full_noM        | nan     | nan     | nan     | nan     | nan     | nan     |
| full_noMY       |   0.369 |   0.347 |   0.374 |   0.354 |   0.357 |   0.283 |
| layer_T         |   0.424 |   0.388 |   0.37  |   0.348 |   0.349 |   0.373 |
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
| full_noMY       | 0.99  |
| layer_T         | 0.952 |
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
| full            | 0.908 |
| full_noM        | 0.908 |
| full_noMY       | 0.932 |
| layer_T         | 0.953 |
| layer_S         | 0.922 |
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
| full_noMY       | 0.036 |
| layer_T         | 0.045 |
| layer_S         | 0     |
| layer_L         | 0     |
| raw_T           | 0     |
| raw_S           | 0.045 |
| copypaste       | 0     |
| account_machine | 0.936 |

## Generator C: h3_ratio@0.01

| detector        |      L3 |
|:----------------|--------:|
| full            |   0.292 |
| full_noM        | nan     |
| full_noMY       |   0.236 |
| layer_T         |   0.341 |
| layer_S         |   0.775 |
| layer_L         | nan     |
| raw_T           |   1.139 |
| raw_S           |   0.729 |
| copypaste       | nan     |
| account_machine | nan     |

