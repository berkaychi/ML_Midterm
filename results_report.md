# Cancer Diagnosis using Blood Microbiome Data - Results Report

This report summarizes the performance of Random Forest and XGBoost classifiers for diagnosing four cancer types based on blood microbiome data, following the methodology outlined in `plan.md`.

## Methodology Overview
- **Data:** `labels.csv`, `data.csv`
- **Preprocessing:** Counts normalized by dividing by the sum per sample.
- **Models:** RandomForestClassifier, XGBClassifier
- **Evaluation:** Stratified 5-Fold Cross-Validation
- **Hyperparameter Tuning:** RandomizedSearchCV (n_iter=50)
- **Metrics:** Sensitivity, Specificity

## Performance Summary

| Task (Cancer vs Others)   | Model        |   Sensitivity |   Specificity |   Training Time (s) |
|:--------------------------|:-------------|--------------:|--------------:|--------------------:|
| colon cancer              | RandomForest |        0.9632 |        1      |               29.27 |
| colon cancer              | XGBoost      |        0.9628 |        0.9878 |              246.25 |
| lung cancer               | RandomForest |        0.95   |        1      |               15.06 |
| lung cancer               | XGBoost      |        0.95   |        0.994  |              184.51 |
| breast cancer             | RandomForest |        0.9524 |        1      |               26.21 |
| breast cancer             | XGBoost      |        0.9905 |        1      |              246.25 |
| prosrtate cancer          | RandomForest |        0.9833 |        0.9741 |               23.67 |
| prosrtate cancer          | XGBoost      |        0.9833 |        0.9871 |              255.13 |

## Best Hyperparameters Found

| Task (Cancer vs Others)   | Model        | Best Params                                                                                                           |
|:--------------------------|:-------------|:----------------------------------------------------------------------------------------------------------------------|
| colon cancer              | RandomForest | {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 30, 'bootstrap': False}             |
| colon cancer              | XGBoost      | {'subsample': 1.0, 'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.2, 'gamma': 0.1, 'colsample_bytree': 1.0}  |
| lung cancer               | RandomForest | {'n_estimators': 300, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': 10, 'bootstrap': True}             |
| lung cancer               | XGBoost      | {'subsample': 0.6, 'n_estimators': 100, 'max_depth': 9, 'learning_rate': 0.1, 'gamma': 0.2, 'colsample_bytree': 0.6}  |
| breast cancer             | RandomForest | {'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_depth': 30, 'bootstrap': False}             |
| breast cancer             | XGBoost      | {'subsample': 0.7, 'n_estimators': 200, 'max_depth': 9, 'learning_rate': 0.05, 'gamma': 0.1, 'colsample_bytree': 0.6} |
| prosrtate cancer          | RandomForest | {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_depth': None, 'bootstrap': False}           |
| prosrtate cancer          | XGBoost      | {'subsample': 0.7, 'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.1, 'gamma': 0.1, 'colsample_bytree': 1.0}  |
