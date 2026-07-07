# CKD Model Comparison — Publication Table

## UCI_Binary_CKD

| model | is_best | cv_roc_auc_mean | cv_roc_auc_std | test_roc_auc | test_f1 | test_mcc | test_sensitivity | test_specificity | calibration_ece | training_time_s | inference_time_ms | n_features | model_size_kb | rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LogisticRegression |  | 0.9994 | 0.0013 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | nan | 0.1300 | 0.2270 | 24 | 1.8000 | 1.0000 |
| RandomForest |  | 0.9988 | 0.0015 | 0.9830 | 0.9901 | 0.9735 | 1.0000 | 0.9667 | nan | 1.3800 | 13.3930 | 24 | 568.3000 | 4.0000 |
| XGBoost |  | 0.9979 | 0.0026 | 0.9830 | 0.9800 | 0.9467 | 0.9800 | 0.9667 | nan | 1.1400 | 0.4180 | 23 | 245.1000 | 4.0000 |
| LightGBM |  | 0.9992 | 0.0010 | 0.9900 | 0.9899 | 0.9739 | 0.9800 | 1.0000 | nan | 1.3700 | 0.4750 | 23 | 200.1000 | 3.0000 |
| CatBoost | ✅ | 0.9996 | 0.0008 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0014 | 0.9200 | 0.3000 | 23 | 333.9000 | 1.0000 |


## Kaggle_Multiclass_CKD_Staging

| model | is_best | cv_balanced_acc_mean | cv_balanced_acc_std | test_balanced_acc | macro_f1 | cohen_kappa | macro_roc_auc | training_time_s | inference_time_ms | n_features | model_size_kb | rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RandomForest | ✅ | 0.8419 | 0.0646 | 0.7810 | 0.7895 | 0.7753 | 0.9528 | 5.9400 | 13.3970 | 34 | 1339.7000 | 1.0000 |
| XGBoost |  | 0.8229 | 0.0652 | 0.7238 | 0.7344 | 0.7097 | 0.9432 | 28.4100 | 0.6850 | 35 | 1310.1000 | 3.0000 |
| LightGBM |  | 0.8106 | 0.0625 | 0.7921 | 0.8086 | 0.7758 | 0.9605 | 21.0400 | 0.6490 | 36 | 859.2000 | 4.0000 |
| CatBoost |  | 0.8409 | 0.0678 | 0.8429 | 0.8565 | 0.8401 | 0.9552 | 4.7100 | 0.3740 | 35 | 931.7000 | 2.0000 |
