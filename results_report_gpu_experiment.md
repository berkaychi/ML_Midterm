# Cancer Diagnosis using Blood Microbiome Data - GPU Experiment Results Report

## 1. Introduction

This report details the outcomes of a project aimed at diagnosing four distinct cancer types (colon, lung, breast, and prostate) utilizing blood microbiome data. The primary methodologies employed include two machine learning classifiers: Random Forest and XGBoost. To optimize model performance, hyperparameter tuning was conducted using `RandomizedSearchCV` with `n_iter_search` set to 50, and model robustness was assessed via Stratified K-Fold Cross-Validation. A significant aspect of this work involved experimentation with GPU optimization for the XGBoost classifier, specifically through the use of `DMatrix` for efficient data handling and `xgb.cv` for cross-validation, with the goal of addressing "mismatched devices" warnings and potentially enhancing computational speed. This document summarizes the comparative performance of these models and the findings from the GPU optimization efforts.

## 2. Methodology Overview

- **Data Source:** The analysis was performed on `labels.csv` (containing sample classifications) and `data.csv` (containing microbiome counts).
- **Data Preprocessing:** Microbiome counts were normalized by dividing each count by the sum of counts per sample, converting them into relative abundances.
- **Machine Learning Models:**
  - `RandomForestClassifier` from scikit-learn.
  - `XGBClassifier` from XGBoost.
- **Hyperparameter Tuning:** `RandomizedSearchCV` was employed with 50 iterations (`n_iter_search=50`) to find the optimal hyperparameter set for each model and cancer type.
- **Model Evaluation:** A Stratified 5-Fold Cross-Validation strategy was used to ensure robust performance estimates, particularly important for imbalanced datasets.
- **Performance Metrics:** Model performance was primarily assessed using Sensitivity (True Positive Rate) and Specificity (True Negative Rate). Training times were also recorded.
- **GPU Optimization (XGBoost):**
  - Data was converted to XGBoost's optimized data structure, `DMatrix`, to facilitate GPU utilization.
  - `xgb.cv` was used for cross-validation with the `gpu_hist` tree method. This approach was specifically implemented to resolve "mismatched devices" warnings encountered in previous experiments and to explore potential speedups in training time.

## 3. Performance Summary

The table below presents the sensitivity, specificity, and training time for Random Forest and XGBoost models across the four cancer types.

| Task (Cancer vs Others) | Model        | Sensitivity | Specificity | Training Time (s) |
| :---------------------- | :----------- | :---------- | :---------- | :---------------- |
| Colon Cancer            | RandomForest | **0.9632**  | **1.0000**  | 28.57             |
| Colon Cancer            | XGBoost      | 0.9537      | 0.9715      | 45.59             |
| Lung Cancer             | RandomForest | **0.9500**  | **1.0000**  | 16.06             |
| Lung Cancer             | XGBoost      | 0.8500      | 0.9911      | 31.58             |
| Breast Cancer           | RandomForest | 0.9524      | **1.0000**  | 27.50             |
| Breast Cancer           | XGBoost      | **0.9810**  | **1.0000**  | 46.59             |
| Prostate Cancer         | RandomForest | **0.9833**  | 0.9741      | 24.76             |
| Prostate Cancer         | XGBoost      | 0.9667      | **0.9871**  | 43.15             |

_Note: "prosrtate cancer" has been corrected to "Prostate Cancer". Highest sensitivity and specificity for each cancer type are highlighted in bold._

## 4. Key Findings and Discussion

This section analyzes the performance of the models and the impact of GPU optimization.

**Model Performance Comparison:**

- **Colon Cancer:** Random Forest demonstrated superior performance with perfect specificity (1.0000) and slightly higher sensitivity (0.9632) compared to XGBoost (Sensitivity: 0.9537, Specificity: 0.9715).
- **Lung Cancer:** Random Forest again outperformed XGBoost, achieving perfect specificity (1.0000) and significantly higher sensitivity (0.9500 vs. 0.8500 for XGBoost).
- **Breast Cancer:** XGBoost showed the highest sensitivity (0.9810), while both models achieved perfect specificity (1.0000). Random Forest's sensitivity was also high at 0.9524.
- **Prostate Cancer:** Random Forest yielded the highest sensitivity (0.9833), while XGBoost achieved slightly better specificity (0.9871 vs. 0.9741 for Random Forest). Both models performed strongly.

**Performance Differences:**
Across most cancer types, Random Forest generally matched or exceeded XGBoost in terms of sensitivity and specificity, particularly for Colon and Lung cancer. For Breast Cancer, XGBoost had a slight edge in sensitivity. For Prostate Cancer, the performance was more comparable, with Random Forest leading in sensitivity and XGBoost in specificity.

**Training Times:**
Random Forest models consistently had shorter training times compared to their XGBoost counterparts across all cancer types. This is an expected outcome, as XGBoost's gradient boosting mechanism is typically more computationally intensive.

**Impact of GPU Optimization (DMatrix and xgb.cv):**
The implementation of `DMatrix` and `xgb.cv` for XGBoost was crucial. As observed in the development process, this approach successfully resolved the "mismatched devices" warning that occurred when attempting GPU acceleration with scikit-learn's wrapper. While direct comparative speedup figures against a non-GPU-optimized XGBoost (that might have failed or run on CPU by default due to the warning) are not the focus of this specific table, the successful execution on GPU using `DMatrix` and `xgb.cv` confirms the viability of this method for leveraging GPU resources. This is a significant methodological improvement for handling larger datasets or more complex XGBoost models where GPU acceleration is desired. The recorded training times for XGBoost reflect GPU-accelerated training.

## 5. Best Hyperparameters Found

Below are the best hyperparameters determined by RandomizedSearchCV for each model and task.

### Colon Cancer

- **RandomForest:**
  - `n_est`: 100
  - `min_spl`: 5
  - `min_leaf`: 1
  - `m_dep`: 30
  - `boot`: False
- **XGBoost:**
  - `sub_s`: 1.0
  - `n_est`: 300
  - `m_dep`: 7
  - `lr`: 0.1
  - `gam`: 0
  - `col_tree`: 1.0
  - `act_n_est`: 3

### Lung Cancer

- **RandomForest:**
  - `n_est`: 300
  - `min_spl`: 10
  - `min_leaf`: 1
  - `m_dep`: 10
  - `boot`: True
- **XGBoost:**
  - `sub_s`: 0.6
  - `n_est`: 300
  - `m_dep`: 7
  - `lr`: 0.05
  - `gam`: 0
  - `col_tree`: 0.6
  - `act_n_est`: 10

### Breast Cancer

- **RandomForest:**
  - `n_est`: 500
  - `min_spl`: 2
  - `min_leaf`: 4
  - `m_dep`: 30
  - `boot`: False
- **XGBoost:**
  - `sub_s`: 0.7
  - `n_est`: 300
  - `m_dep`: 7
  - `lr`: 0.2
  - `gam`: 0.1
  - `col_tree`: 0.8
  - `act_n_est`: 13

### Prostate Cancer

- **RandomForest:**
  - `n_est`: 100
  - `min_spl`: 5
  - `min_leaf`: 4
  - `m_dep`: None
  - `boot`: False
- **XGBoost:**
  - `sub_s`: 0.6
  - `n_est`: 300
  - `m_dep`: 5
  - `lr`: 0.2
  - `gam`: 0
  - `col_tree`: 1.0
  - `act_n_est`: 31

## 6. Conclusion

This study successfully evaluated Random Forest and XGBoost classifiers for cancer diagnosis using blood microbiome data, incorporating GPU optimization techniques for XGBoost.

**Key Findings:**

- Random Forest generally provided robust performance, often achieving the highest sensitivity and/or specificity, particularly for Colon and Lung cancer, with the added benefit of faster training times.
- XGBoost, while slightly more time-consuming to train, showed competitive results, especially for Breast Cancer (highest sensitivity) and Prostate Cancer (highest specificity).
- The use of `DMatrix` and `xgb.cv` was effective in enabling GPU-accelerated training for XGBoost and resolved prior "mismatched devices" issues, demonstrating a practical approach for GPU utilization in XGBoost workflows.

**Promising Model Approaches:**

- For Colon and Lung cancer diagnosis, Random Forest appears to be a highly effective and efficient model.
- For Breast Cancer, XGBoost showed a slight advantage in sensitivity.
- For Prostate Cancer, both models performed well, with Random Forest leading in sensitivity and XGBoost in specificity, suggesting that the choice might depend on whether sensitivity or specificity is prioritized.

**Future Directions:**

- **Feature Importance Analysis:** Investigate the most influential microbial taxa for each cancer type, which could provide biological insights.
- **Advanced Deep Learning Models:** Explore convolutional neural networks (CNNs) or other deep learning architectures, which might capture more complex patterns in the microbiome data, especially with larger datasets.
- **Ensemble Methods:** Experiment with stacking or blending Random Forest and XGBoost models to potentially achieve further performance gains.
- **Larger Datasets:** Validate these findings on larger and more diverse patient cohorts to ensure generalizability.
- **Prospective Studies:** Ultimately, prospective clinical studies would be needed to confirm the diagnostic utility of these microbiome-based classifiers.

This research underscores the potential of blood microbiome data in cancer diagnostics and highlights the practical considerations for model selection and optimization, including leveraging GPU capabilities.
