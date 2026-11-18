# Exploration summary: `day_top3`

Runs: 31 files. Mean over seeds (generator A); generator C (unseen) shown separately.

## Generator A: recall@0.001

| detector        |    L0 |    L1 |    L2 |    L3 |    L4 |    L5 |
|:----------------|------:|------:|------:|------:|------:|------:|
| full            | 1     | 1     | 1     | 0.993 | 0.548 | 0.624 |
| gnn_all         | 0.995 | 1     | 1     | 1     | 0.976 | 0.981 |
| gnn_style       | 0.995 | 1     | 1     | 1     | 0.586 | 0.743 |
| gnn_none        | 0.995 | 1     | 1     | 1     | 0.562 | 0.657 |
| full_noM        | 1     | 1     | 1     | 0.993 | 0.545 | 0.605 |
| full_noMY       | 1     | 1     | 1     | 0.988 | 0.443 | 0.538 |
| layer_T         | 0.967 | 0.967 | 0.976 | 0.969 | 0.357 | 0.324 |
| layer_S         | 1     | 1     | 0.998 | 0.967 | 0.529 | 0.593 |
| layer_L         | 1     | 1     | 0     | 0.019 | 0     | 0.007 |
| raw_T           | 0.381 | 0.381 | 0.381 | 0.381 | 0.055 | 0     |
| raw_S           | 1     | 1     | 0.995 | 0.933 | 0.498 | 0.574 |
| copypaste       | 1     | 1     | 0     | 0.002 | 0     | 0.01  |
| account_machine | 0.681 | 0.762 | 0.714 | 0.514 | 0.438 | 0.371 |
| luceri          | 0.595 | 0.6   | 0.512 | 0.255 | 0.048 | 0.16  |

## Generator A: recall@0.01

| detector        |    L0 |    L1 |    L2 |    L3 |    L4 |    L5 |
|:----------------|------:|------:|------:|------:|------:|------:|
| full            | 1     | 1     | 1     | 1     | 0.633 | 0.721 |
| gnn_all         | 0.995 | 1     | 1     | 1     | 1     | 1     |
| gnn_style       | 0.995 | 1     | 1     | 1     | 0.629 | 0.862 |
| gnn_none        | 0.995 | 1     | 1     | 1     | 0.605 | 0.657 |
| full_noM        | 1     | 1     | 1     | 1     | 0.633 | 0.719 |
| full_noMY       | 1     | 1     | 1     | 1     | 0.581 | 0.698 |
| layer_T         | 0.993 | 0.993 | 0.995 | 0.995 | 0.462 | 0.56  |
| layer_S         | 1     | 1     | 0.998 | 0.967 | 0.529 | 0.593 |
| layer_L         | 1     | 1     | 0     | 0.019 | 0     | 0.007 |
| raw_T           | 0.795 | 0.795 | 0.795 | 0.795 | 0.26  | 0.231 |
| raw_S           | 1     | 1     | 1     | 0.99  | 0.69  | 0.793 |
| copypaste       | 1     | 1     | 0.019 | 0.017 | 0     | 0.014 |
| account_machine | 1     | 1     | 1     | 1     | 1     | 1     |
| luceri          | 0.845 | 0.848 | 0.8   | 0.71  | 0.205 | 0.457 |

## Generator A: auprc

| detector        |    L0 |    L1 |    L2 |    L3 |    L4 |    L5 |
|:----------------|------:|------:|------:|------:|------:|------:|
| full            | 0.99  | 0.991 | 0.938 | 0.913 | 0.501 | 0.584 |
| gnn_all         | 0.995 | 0.995 | 1     | 0.99  | 0.939 | 0.946 |
| gnn_style       | 0.995 | 0.995 | 0.994 | 1     | 0.613 | 0.799 |
| gnn_none        | 0.995 | 0.995 | 0.981 | 0.974 | 0.567 | 0.659 |
| full_noM        | 0.989 | 0.991 | 0.936 | 0.911 | 0.496 | 0.575 |
| full_noMY       | 0.951 | 0.953 | 0.937 | 0.928 | 0.449 | 0.547 |
| layer_T         | 0.977 | 0.978 | 0.977 | 0.976 | 0.386 | 0.408 |
| layer_S         | 0.941 | 0.941 | 0.934 | 0.893 | 0.438 | 0.506 |
| layer_L         | 1     | 1     | 0.007 | 0.017 | 0.007 | 0.012 |
| raw_T           | 0.557 | 0.557 | 0.556 | 0.555 | 0.115 | 0.105 |
| raw_S           | 0.993 | 0.993 | 0.985 | 0.935 | 0.538 | 0.624 |
| copypaste       | 1     | 1     | 0.009 | 0.009 | 0.007 | 0.013 |
| account_machine | 0.723 | 0.752 | 0.723 | 0.663 | 0.625 | 0.578 |
| luceri          | 0.691 | 0.705 | 0.594 | 0.429 | 0.111 | 0.26  |

## Generator A: benign_flag@0.01

| detector        |    L0 |    L1 |    L2 |    L3 |    L4 |    L5 |
|:----------------|------:|------:|------:|------:|------:|------:|
| full            | 0.008 | 0.008 | 0.008 | 0.008 | 0.008 | 0.004 |
| gnn_all         | 0     | 0     | 0.004 | 0.008 | 0.212 | 0.223 |
| gnn_style       | 0     | 0     | 0.004 | 0.008 | 0.112 | 0.4   |
| gnn_none        | 0     | 0     | 0.004 | 0     | 0.012 | 0.012 |
| full_noM        | 0     | 0     | 0     | 0.004 | 0     | 0.004 |
| full_noMY       | 0.01  | 0.006 | 0.004 | 0.008 | 0.01  | 0.008 |
| layer_T         | 0.01  | 0.006 | 0.01  | 0.008 | 0.01  | 0.008 |
| layer_S         | 0     | 0     | 0.002 | 0     | 0     | 0     |
| layer_L         | 0     | 0     | 0     | 0     | 0     | 0     |
| raw_T           | 0     | 0     | 0     | 0     | 0     | 0     |
| raw_S           | 0.025 | 0.025 | 0.027 | 0.029 | 0.033 | 0.037 |
| copypaste       | 0     | 0     | 0     | 0     | 0     | 0     |
| account_machine | 0.223 | 0.223 | 0.223 | 0.223 | 0.223 | 0.223 |
| luceri          | 0     | 0     | 0     | 0     | 0     | 0     |

## Generator A: h3_ratio@0.01

| detector        |      L0 |      L1 |      L2 |      L3 |      L4 |      L5 |
|:----------------|--------:|--------:|--------:|--------:|--------:|--------:|
| full            |   0.367 |   0.356 |   0.356 |   0.356 |   0.337 |   0.328 |
| gnn_all         | nan     | nan     | nan     | nan     | nan     | nan     |
| gnn_style       | nan     | nan     | nan     | nan     | nan     | nan     |
| gnn_none        | nan     | nan     | nan     | nan     | nan     | nan     |
| full_noM        | nan     | nan     | nan     | nan     | nan     | nan     |
| full_noMY       |   0.352 |   0.363 |   0.332 |   0.345 |   0.306 |   0.318 |
| layer_T         |   0.342 |   0.349 |   0.323 |   0.335 |   0.294 |   0.325 |
| layer_S         |   0.784 |   0.784 |   0.784 |   0.784 |   0.851 |   0.767 |
| layer_L         | nan     | nan     | nan     | nan     | nan     | nan     |
| raw_T           |   1.187 |   1.187 |   1.187 |   1.188 |   1.39  |   1.404 |
| raw_S           |   0.748 |   0.751 |   0.745 |   0.759 |   0.758 |   0.706 |
| copypaste       | nan     | nan     | nan     | nan     | nan     | nan     |
| account_machine | nan     | nan     | nan     | nan     | nan     | nan     |
| luceri          | nan     | nan     | nan     | nan     | nan     | nan     |

## Generator C: recall@0.001

| detector        |    L3 |
|:----------------|------:|
| full            | 1     |
| full_noM        | 1     |
| full_noMY       | 0.99  |
| layer_T         | 0.971 |
| layer_S         | 0.99  |
| layer_L         | 0     |
| raw_T           | 0.481 |
| raw_S           | 0.971 |
| copypaste       | 0.1   |
| account_machine | 0.062 |
| luceri          | 0.5   |

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
| luceri          | 0.762 |

## Generator C: auprc

| detector        |    L3 |
|:----------------|------:|
| full            | 0.926 |
| full_noM        | 0.926 |
| full_noMY       | 0.935 |
| layer_T         | 0.982 |
| layer_S         | 0.925 |
| layer_L         | 0.007 |
| raw_T           | 0.558 |
| raw_S           | 0.973 |
| copypaste       | 0.198 |
| account_machine | 0.449 |
| luceri          | 0.561 |

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
| luceri          | 0     |

## Generator C: h3_ratio@0.01

| detector        |      L3 |
|:----------------|--------:|
| full            |   0.332 |
| full_noM        | nan     |
| full_noMY       |   0.373 |
| layer_T         |   0.332 |
| layer_S         |   0.775 |
| layer_L         | nan     |
| raw_T           |   1.139 |
| raw_S           |   0.729 |
| copypaste       | nan     |
| account_machine | nan     |
| luceri          | nan     |

