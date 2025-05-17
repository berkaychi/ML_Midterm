# Cancer Diagnosis using Blood Microbiome Data - Results Report

This report summarizes the performance of Random Forest and XGBoost classifiers for diagnosing four cancer types based on blood microbiome data, following the methodology outlined in `plan.md`.

## Methodology Overview
- **Data:** `labels.csv`, `data.csv`
- **Preprocessing:** Counts normalized by dividing by the sum per sample.
- **Models:** RandomForestClassifier, XGBClassifier
- **Evaluation:** Stratified 5-Fold Cross-Validation
- **Hyperparameter Tuning:** RandomizedSearchCV (n_iter=5)
- **Metrics:** Sensitivity, Specificity

## Performance Summary

| Task (Cancer vs Others)   | Model        |   Sensitivity |   Specificity |   Training Time (s) |
|:--------------------------|:-------------|--------------:|--------------:|--------------------:|
| colon cancer              | RandomForest |        0.9632 |        1      |               10.31 |
| colon cancer              | XGBoost      |        0.9532 |        0.9918 |               50.87 |
| lung cancer               | RandomForest |        0.9    |        1      |                2.74 |
| lung cancer               | XGBoost      |        0.95   |        0.994  |               37.52 |
| breast cancer             | RandomForest |        0.9242 |        1      |                4.93 |
| breast cancer             | XGBoost      |        0.981  |        0.996  |               54.04 |
| prosrtate cancer          | RandomForest |        0.975  |        0.9957 |                4.47 |
| prosrtate cancer          | XGBoost      |        0.975  |        0.9871 |               48.59 |

## Best Hyperparameters Found

| Task (Cancer vs Others)   | Model        | Best Params                                                                                                          |
|:--------------------------|:-------------|:---------------------------------------------------------------------------------------------------------------------|
| colon cancer              | RandomForest | {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 30, 'bootstrap': False}            |
| colon cancer              | XGBoost      | {'subsample': 0.6, 'n_estimators': 100, 'max_depth': 9, 'learning_rate': 0.1, 'gamma': 0.2, 'colsample_bytree': 0.6} |
| lung cancer               | RandomForest | {'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': 10, 'bootstrap': True}            |
| lung cancer               | XGBoost      | {'subsample': 0.6, 'n_estimators': 100, 'max_depth': 9, 'learning_rate': 0.1, 'gamma': 0.2, 'colsample_bytree': 0.6} |
| breast cancer             | RandomForest | {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_depth': None, 'bootstrap': False}          |
| breast cancer             | XGBoost      | {'subsample': 0.7, 'n_estimators': 300, 'max_depth': 9, 'learning_rate': 0.01, 'gamma': 0, 'colsample_bytree': 1.0}  |
| prosrtate cancer          | RandomForest | {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_depth': None, 'bootstrap': False}          |
| prosrtate cancer          | XGBoost      | {'subsample': 0.6, 'n_estimators': 100, 'max_depth': 9, 'learning_rate': 0.1, 'gamma': 0.2, 'colsample_bytree': 0.6} |
