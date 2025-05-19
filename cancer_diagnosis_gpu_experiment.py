import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, ParameterSampler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, make_scorer
import time


xgboost_selected_tree_method = 'hist'
xgboost_device_setting = 'cpu'
xgboost_device_message = "XGBoost will run on CPU (device='cpu', tree_method='hist'). Reason: Default fallback, check failed or not attempted."

try:
    _check_xgb = xgb.XGBClassifier(device='cuda', tree_method='hist', n_estimators=1, eval_metric='logloss', random_state=42)
    _dummy_X_check = np.array([[0,0],[1,1]])
    _dummy_y_check = np.array([0,1])
    _check_xgb.fit(_dummy_X_check, _dummy_y_check)
    
    # If fit succeeds, GPU is usable
    xgboost_device_setting = 'cuda'
    xgboost_selected_tree_method = 'hist'
    xgboost_device_message = "XGBoost GPU is available and configured. Using device='cuda', tree_method='hist'."
    print(xgboost_device_message)
except Exception as e:
    xgboost_device_message = f"XGBoost GPU not available or setup error (e.g., not compiled with GPU support, no driver): {str(e)}. Falling back to CPU (device='cpu', tree_method='hist')."
    print(xgboost_device_message)
    xgboost_device_setting = 'cpu'
    xgboost_selected_tree_method = 'hist'



def sensitivity(y_true, y_pred, positive_label=1):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def specificity(y_true, y_pred, positive_label=1):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0


scorers = {
    'sensitivity': make_scorer(sensitivity),
    'specificity': make_scorer(specificity)
}


print("Loading data...")
try:
    # Load labels
    labels_df = pd.read_csv('labels.csv')
    print(f"Labels loaded: {labels_df.shape[0]} rows, {labels_df.shape[1]} columns")
    print("Labels columns:", labels_df.columns.tolist())
    print("Labels head:\n", labels_df.head())

    # Load features (microbiome data)
    data_df = pd.read_csv('data.csv')
    print(f"\nData loaded: {data_df.shape[0]} rows, {data_df.shape[1]} columns")
    print("Data columns preview:", data_df.columns[:5].tolist(), "...") # Show only first few
    print("Data head:\n", data_df.head())

    label_id_col = labels_df.columns[0]
    data_id_col = data_df.columns[0]
    if label_id_col != data_id_col:
        print(f"\nWarning: ID columns might differ ('{label_id_col}' vs '{data_id_col}'). Assuming first column is the sample ID for merging.")
        print(f"Renaming '{data_id_col}' to '{label_id_col}' in data_df for merging.")
        data_df.rename(columns={data_id_col: label_id_col}, inplace=True)

    # Merge dataframes
    print(f"\nMerging dataframes on '{label_id_col}'...")
    merged_df = pd.merge(labels_df, data_df, on=label_id_col, how='inner')
    print(f"Merged data shape: {merged_df.shape}")

    # Initial Exploration
    print("\n--- Initial Exploration ---")
    print("Merged data info:")
    merged_df.info()

    print("\nChecking for missing values:")
    print(merged_df.isnull().sum().sort_values(ascending=False))
    # Check if any column has missing values
    if merged_df.isnull().any().any():
        print("\nWarning: Missing values detected!")
    else:
        print("No missing values found.")


    target_col = labels_df.columns[1]
    print(f"\nDistribution of target variable ('{target_col}'):")
    print(merged_df[target_col].value_counts())

except FileNotFoundError as e:
    print(f"Error: {e}. Make sure 'labels.csv' and 'data.csv' are in the directory.")
    merged_df = None
except Exception as e:
    merged_df = None
    print(f"An error occurred: {e}")

# Data Preprocessing
print("\n--- Step 3: Data Preprocessing ---")
if 'merged_df' in locals() and merged_df is not None:
    feature_columns = merged_df.columns[2:]
    print(f"Identified {len(feature_columns)} feature columns starting from '{feature_columns[0]}'.")

    X = merged_df[feature_columns]
    y = merged_df[target_col]

    print("Applying normalization (dividing counts by row sum)...")
    row_sums = X.sum(axis=1)
    row_sums[row_sums == 0] = 1
    X_normalized = X.div(row_sums, axis=0)

    print("Normalization complete.")
    print("Normalized features shape:", X_normalized.shape)
    print("Normalized features head:\n", X_normalized.head())

    # Define Binary Classification Tasks
    print("\n--- Step 4: Define Binary Classification Tasks ---")
    cancer_types = y.unique()
    print(f"Unique cancer types found: {cancer_types}")

    binary_targets = {}
    for cancer_type in cancer_types:
        binary_targets[cancer_type] = y.apply(lambda x: 1 if x == cancer_type else 0)
        print(f"Created binary target for '{cancer_type}':")
        print(binary_targets[cancer_type].value_counts())

    print("\nBinary target variables created.")

    print("\n--- Step 5: Model Training & Evaluation ---")
    results = {}
    n_folds = 5
    cv_strategy = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    n_iter_search = 50
    # --- Define Parameter Grids for RandomizedSearchCV ---
    # Random Forest parameters
    rf_param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # XGBoost parameters
    xgb_param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3]
    }

    # --- Loop through each classification task ---
    for cancer_type, y_binary in binary_targets.items():
        print(f"\n===== Training for: {cancer_type} vs Others =====")
        task_results = {}

        # --- Random Forest ---
        print(f"\n--- Running RandomizedSearchCV for Random Forest ---")
        start_time = time.time()
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        rf_random_search = RandomizedSearchCV(
            estimator=rf, param_distributions=rf_param_dist, n_iter=n_iter_search,
            cv=cv_strategy, scoring=scorers, refit='sensitivity',
            n_jobs=-1, random_state=42, verbose=1
        )
        rf_random_search.fit(X_normalized.values, y_binary)
        task_results['RandomForest'] = {
            'Sensitivity': rf_random_search.cv_results_['mean_test_sensitivity'][rf_random_search.best_index_],
            'Specificity': rf_random_search.cv_results_['mean_test_specificity'][rf_random_search.best_index_],
            'Best Params': rf_random_search.best_params_,
            'Training Time (s)': time.time() - start_time
        }
        print(f"RF Best Sensitivity: {task_results['RandomForest']['Sensitivity']:.4f}")
        print(f"RF Best Specificity: {task_results['RandomForest']['Specificity']:.4f}")

        # --- XGBoost with xgb.cv and DMatrix ---
        print(f"\n--- Running Custom Hyperparameter Search for XGBoost with xgb.cv ---")
        start_time_xgb_total = time.time()

        # Prepare DMatrix
        X_np = X_normalized.values if isinstance(X_normalized, pd.DataFrame) else X_normalized
        dtrain_xgb = xgb.DMatrix(X_np, label=y_binary)

        best_xgb_sensitivity = -1
        best_xgb_specificity = -1
        best_xgb_params = None
        
        def xgb_eval_metrics_cv(preds, dtrain):
            labels = dtrain.get_label()
            pred_labels = (preds > 0.5).astype(int)
            sens = sensitivity(labels, pred_labels)
            spec = specificity(labels, pred_labels)
            return [('sensitivity', sens), ('specificity', spec)]

        param_sampler = ParameterSampler(xgb_param_dist, n_iter=n_iter_search, random_state=42)

        for i, xgb_params_sampled in enumerate(param_sampler):
            print(f"XGBoost Iteration {i+1}/{n_iter_search}")
            current_params_for_cv = xgb_params_sampled.copy()
            
            current_params_for_cv['objective'] = 'binary:logistic'
            current_params_for_cv['device'] = xgboost_device_setting
            current_params_for_cv['tree_method'] = xgboost_selected_tree_method
            current_params_for_cv['eval_metric'] = 'logloss'
            current_params_for_cv['seed'] = 42
            
            if sum(y_binary==1) > 0:
                current_params_for_cv['scale_pos_weight'] = sum(y_binary==0) / sum(y_binary==1)
            else:
                current_params_for_cv['scale_pos_weight'] = 1


            num_boost_round_sampled = current_params_for_cv.pop('n_estimators', 200)

            try:
                cv_results = xgb.cv(
                    params=current_params_for_cv,
                    dtrain=dtrain_xgb,
                    num_boost_round=num_boost_round_sampled,
                    folds=cv_strategy,
                    custom_metric=xgb_eval_metrics_cv,
                    maximize=True,
                    early_stopping_rounds=10,
                    verbose_eval=False
                )
                
                current_sensitivity = cv_results['test-sensitivity-mean'].iloc[-1]
                current_specificity = cv_results['test-specificity-mean'].iloc[-1]
                actual_n_estimators = len(cv_results)

                print(f"  Params: {xgb_params_sampled}, Actual Estimators: {actual_n_estimators}, Sensitivity: {current_sensitivity:.4f}, Specificity: {current_specificity:.4f}")

                if current_sensitivity > best_xgb_sensitivity:
                    best_xgb_sensitivity = current_sensitivity
                    best_xgb_specificity = current_specificity
                    best_xgb_params = xgb_params_sampled.copy()
                    best_xgb_params['actual_n_estimators'] = actual_n_estimators


            except Exception as e_cv:
                print(f"Error during xgb.cv with params {current_params_for_cv}: {e_cv}")
                continue
        
        if best_xgb_params:
            task_results['XGBoost'] = {
                'Sensitivity': best_xgb_sensitivity,
                'Specificity': best_xgb_specificity,
                'Best Params': best_xgb_params,
                'Training Time (s)': time.time() - start_time_xgb_total
            }
            print(f"XGB Best Sensitivity (from xgb.cv): {best_xgb_sensitivity:.4f}")
            print(f"XGB Best Specificity (from xgb.cv): {best_xgb_specificity:.4f}")
            print(f"XGB Best Params: {best_xgb_params}")
        else:
            task_results['XGBoost'] = {
                'Sensitivity': 0,
                'Specificity': 0,
                'Best Params': {},
                'Training Time (s)': time.time() - start_time_xgb_total
            }
            print("XGBoost training with xgb.cv failed to find best parameters.")

        results[cancer_type] = task_results

    # Results Compilation
    print("\n--- Step 6: Results Compilation ---")
    summary_data = []
    for cancer_type, task_res in results.items():
        for model_name, metrics in task_res.items():
            summary_data.append({
                'Task (Cancer vs Others)': cancer_type,
                'Model': model_name,
                'Sensitivity': f"{metrics['Sensitivity']:.4f}",
                'Specificity': f"{metrics['Specificity']:.4f}",
                'Training Time (s)': f"{metrics['Training Time (s)']:.2f}",
                'Best Params': metrics['Best Params']
            })

    results_df = pd.DataFrame(summary_data)
    print("Compiled Results Summary:")
    print(results_df[['Task (Cancer vs Others)', 'Model', 'Sensitivity', 'Specificity', 'Training Time (s)']])

    # Reporting
    print("\n--- Step 7: Reporting ---")
    report_path = 'results_report_gpu_experiment.md'
    try:
        with open(report_path, 'w') as f:
            f.write("# Cancer Diagnosis using Blood Microbiome Data - Results Report (GPU Experiment)\n\n")
            f.write("This report summarizes the performance of Random Forest and XGBoost classifiers for diagnosing four cancer types based on blood microbiome data, following the methodology outlined in `plan.md`.\n\n")
            f.write("This version includes experiments for GPU optimization.\n\n")
            f.write("## Methodology Overview\n")
            f.write("- **Data:** `labels.csv`, `data.csv`\n")
            f.write("- **Preprocessing:** Counts normalized by dividing by the sum per sample.\n")
            f.write("- **Models:** RandomForestClassifier, XGBClassifier\n")
            f.write("- **Evaluation:** Stratified 5-Fold Cross-Validation\n")
            f.write(f"- **Hyperparameter Tuning:** RandomizedSearchCV (n_iter={n_iter_search})\n")
            f.write("- **Metrics:** Sensitivity, Specificity\n\n")

            f.write("## Performance Summary\n\n")
            results_table_md = results_df[['Task (Cancer vs Others)', 'Model', 'Sensitivity', 'Specificity', 'Training Time (s)']].to_markdown(index=False)
            f.write(results_table_md)
            f.write("\n\n")

            f.write("## Best Hyperparameters Found\n\n")
            f.write(results_df[['Task (Cancer vs Others)', 'Model', 'Best Params']].to_markdown(index=False))
            f.write("\n")
        print(f"Results report successfully saved to {report_path}")
    except Exception as e:
        print(f"Error writing report file: {e}")

else:
    print("\nExecution halted due to errors in data loading or merging.")