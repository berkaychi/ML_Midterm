# Plan: Cancer Diagnosis using Blood Microbiome Data

This plan outlines the steps to diagnose four types of cancer (Colon, Breast, Lung, Prostate) using blood microbiome data, based on the requirements in `ML_odev.pdf`.

## 1. Data Acquisition & Verification

- Confirm `labels.csv` and `data.csv` are present in the workspace (`c:/Users/berka/Desktop/machine_learning`).

## 2. Data Loading & Initial Exploration

- Load `labels.csv` into a pandas DataFrame.
- Load `data.csv` into a pandas DataFrame.
- Merge the two DataFrames using the common sample identifier.
- Explore the merged data: check dimensions, data types, missing values, and the distribution of cancer types (target variable).

## 3. Data Preprocessing

- **Normalization:** Apply the normalization method suggested in the PDF: For each sample (row in `data.csv`), divide each microorganism count by the sum of all counts for that sample.
- **Feature Selection:** Keep all 1836 microorganism features as specified.
- **Target Encoding:** Prepare the target variable (`DiseaseType`) for the classification tasks.

## 4. Define Binary Classification Tasks

- Create four separate binary target variables based on `DiseaseType`:
  - Task 1: Colon Cancer (1) vs. Others (0)
  - Task 2: Breast Cancer (1) vs. Others (0)
  - Task 3: Lung Cancer (1) vs. Others (0)
  - Task 4: Prostate Cancer (1) vs. Others (0)

## 5. Model Training & Evaluation (Stratified 5-Fold Cross-Validation)

- Use 5-fold Stratified Cross-Validation for all evaluations to ensure robust performance metrics.
- For each of the 4 tasks:
  - **Random Forest:**
    - Initialize a Random Forest Classifier.
    - Perform hyperparameter tuning (e.g., using `RandomizedSearchCV` for efficiency) focusing on parameters like `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`.
    - Train the best model on the full dataset (or use cross-validated predictions).
    - Calculate average Sensitivity and Specificity across the 5 folds.
  - **XGBoost (Gradient Boosted Trees):**
    - Initialize an XGBoost Classifier.
    - Perform hyperparameter tuning (e.g., using `RandomizedSearchCV`) focusing on parameters like `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`.
    - Train the best model on the full dataset (or use cross-validated predictions).
    - Calculate average Sensitivity and Specificity across the 5 folds.

## 6. Results Compilation & Comparison

- Create a table summarizing the average Sensitivity and Specificity for each of the 4 tasks, comparing Random Forest and XGBoost side-by-side.

## 7. Reporting

- Generate a `results_report.md` file containing:
  - A brief overview of the task and methodology.
  - The results table.
  - A comparison and discussion of the models' performance for each cancer type.

## Workflow Diagram

```mermaid
graph TD
    A[Start: Load Data (labels.csv, data.csv)] --> B(Merge DataFrames);
    B --> C{Explore Data};
    C --> D[Preprocess Data: Normalize Counts];
    D --> E[Define 4 Binary Classification Tasks];
    E --> F{Task 1: Colon vs Others};
    E --> G{Task 2: Breast vs Others};
    E --> H{Task 3: Lung vs Others};
    E --> I{Task 4: Prostate vs Others};

    subgraph Cross-Validation Loop (for each Task)
        direction LR
        J[Split: Stratified 5-Fold CV] --> K(Train/Tune Random Forest);
        J --> L(Train/Tune XGBoost);
        K --> M[Calculate RF Metrics (Sensitivity, Specificity)];
        L --> N[Calculate XGB Metrics (Sensitivity, Specificity)];
    end

    F --> J;
    G --> J;
    H --> J;
    I --> J;

    M --> O(Compile Results);
    N --> O;

    O --> P[Generate Markdown Report];
    P --> Q[End];
```
