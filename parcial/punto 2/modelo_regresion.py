import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import squarify
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from scipy.stats import mode, ttest_ind, chi2_contingency
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from matplotlib.colors import ListedColormap

# --- Function Definitions ---

# Helper function to calculate KS statistic
def calculate_ks(y_true, y_pred_proba):
    """Calculates the Kolmogorov-Smirnov statistic."""
    df = pd.DataFrame({'y_true': y_true, 'y_pred_proba': y_pred_proba})

    df_bad = df[df['y_true'] == 1]
    df_good = df[df['y_true'] == 0]

    ks_stats = []
    thresholds = sorted(df['y_pred_proba'].unique())

    for threshold in thresholds:
        tpr = len(df_bad[df_bad['y_pred_proba'] >= threshold]) / len(df_bad) if len(df_bad) > 0 else 0
        fpr = len(df_good[df_good['y_pred_proba'] >= threshold]) / len(df_good) if len(df_good) > 0 else 0
        ks_stats.append(abs(tpr - fpr))

    return max(ks_stats) if ks_stats else 0

def run_logistic_regression(df, target_column, test_size=0.3, random_state=42):
    """
    Runs a logistic regression model on the provided DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing features and target variable.
        target_column (str): The name of the target variable column.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before splitting.

    Returns:
        tuple: Contains the trained model, accuracy score, AUC score, KS statistic,
               confusion matrix, classification report, and the test target variable (y_test).
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Initialize and train the Logistic Regression model
    model = LogisticRegression(random_state=random_state, max_iter=1000, solver='newton-cholesky')
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilities for the positive class

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    ks = calculate_ks(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\nLogistic Regression Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"KS Statistic: {ks:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    return model, accuracy, auc, ks, cm, report, y_test # Return y_test

# Helper function to group features based on common prefixes
def get_feature_groups(feature_names):
    groups = {}
    # Define common prefixes used in one-hot encoding
    prefixes = ['BMI_Category_', 'Age_', 'Education_', 'Income_', 'GenHlth_', 'MentHlth_Bin_', 'PhysHlth_Bin_']

    remaining_features = list(feature_names)

    for prefix in prefixes:
        group_name = prefix.rstrip('_')
        group_cols = [f for f in remaining_features if f.startswith(prefix)]
        if group_cols:
            groups[group_name] = group_cols
            # Remove grouped features from the list to process
            remaining_features = [f for f in remaining_features if f not in group_cols]

    # Any feature left is its own group
    for feature in remaining_features:
        groups[feature] = [feature]

    return groups

# Minimal logistic regression runner for internal use in sensitivity analysis
def run_logistic_regression_minimal(X_train, y_train, X_test, y_test, random_state=42):

    model = LogisticRegression(random_state=random_state, max_iter=1000, solver='newton-cholesky')
    # Fit on original data
    model.fit(X_train, y_train)
    # Predict on original data
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilities needed for AUC/KS

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    ks = calculate_ks(y_test, y_pred_proba)

    # Return model and metrics
    return model, accuracy, auc, ks

# Main sensitivity analysis function
def sensitivity_analysis_logistic_regression(df, target_column, random_state=42):
    """
    Performs sensitivity analysis by iteratively removing feature groups
    based on logistic regression coefficient magnitudes.
    Identifies and returns the model with the highest accuracy during the process,
    along with the test data.

    Args:
        df (pd.DataFrame): DataFrame containing features and target variable.
        target_column (str): The name of the target variable column.
        random_state (int): Controls the shuffling and model training.

    Returns:
        tuple: (results_df, fig, best_model, best_model_features, X_test_data, y_test_data, best_removed_group)
            results_df (pd.DataFrame): DataFrame tracking removed groups, accuracy, AUC, and KS.
            fig (matplotlib.figure.Figure): Figure object for the accuracy plot.
            best_model (LogisticRegression): The trained model instance with the highest accuracy.
            best_model_features (list): List of feature names used by the best model.
            X_test_data (pd.DataFrame): The feature data from the test split.
            y_test_data (pd.Series): The target data from the test split.
            best_removed_group (str): The name of the best removed group.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    feature_names = X.columns.tolist()

    # 1. Initial Data Split (consistent across iterations)
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )

    # 2. Train Initial Model
    print("Training initial model for baseline and feature importance...")
    # Updated call to minimal runner
    initial_model, initial_accuracy, initial_auc, initial_ks = run_logistic_regression_minimal(
        X_train_full, y_train, X_test_full, y_test, random_state=random_state
    )
    print(f"Initial Accuracy: {initial_accuracy:.4f}")
    print(f"Initial AUC: {initial_auc:.4f}")
    print(f"Initial KS: {initial_ks:.4f}")


    # Store models and their features/metrics at each step
    models_history = {'None': {'model': initial_model, 'features': list(feature_names), 'accuracy': initial_accuracy, 'auc': initial_auc, 'ks': initial_ks}}
    results = {'Removed Group': ['None'], 'Accuracy': [initial_accuracy], 'AUC': [initial_auc], 'KS': [initial_ks]}

    if not hasattr(initial_model, 'coef_'):
         print("Error: Could not retrieve coefficients from the initial model.")
         # Return None for test data as well
         return pd.DataFrame(results), None, None, [], None, None, None #

    # 3. Group Features
    feature_groups = get_feature_groups(feature_names)
    print(f"Identified {len(feature_groups)} feature groups.")

    # 4. Calculate Group Importance from Initial Model
    initial_coeffs = initial_model.coef_[0]

    feature_importance_map = dict(zip(feature_names, np.abs(initial_coeffs)))

    group_importances = {}
    for group_name, group_cols in feature_groups.items():
        importance = sum(feature_importance_map.get(col, 0) for col in group_cols)
        group_importances[group_name] = importance

    # 5. Sort Groups by Importance (Ascending - least important first)
    sorted_groups = sorted(group_importances.items(), key=lambda item: item[1])

    # 6. Iterative Removal and Retraining
    current_feature_names = list(feature_names)

    print("\nStarting iterative feature group removal (least important first)...")

    for i in range(len(sorted_groups) - 1):
        group_name_to_remove, group_importance = sorted_groups[i]
        cols_to_remove = feature_groups[group_name_to_remove]

        print(f"Step {i+1}: Removing group '{group_name_to_remove}' (Abs Coeff Sum Importance: {group_importance:.4f}, Features: {len(cols_to_remove)})")

        current_feature_names = [f for f in current_feature_names if f not in cols_to_remove]

        if not current_feature_names:
            print("  No features left after removal. Stopping analysis.")
            break

        remaining_X_train = X_train_full[current_feature_names]
        remaining_X_test = X_test_full[current_feature_names]

        # Re-train model and get metrics
        # Updated call to minimal runner
        model, accuracy, auc, ks = run_logistic_regression_minimal(
            remaining_X_train, y_train, remaining_X_test, y_test, random_state=random_state
        )

        print(f"  Accuracy after removal: {accuracy:.4f}")
        print(f"  AUC after removal: {auc:.4f}")
        print(f"  KS after removal: {ks:.4f}")


        # Store result
        results['Removed Group'].append(group_name_to_remove)
        results['Accuracy'].append(accuracy)
        results['AUC'].append(auc)
        results['KS'].append(ks)
        # Store the model, its features, and metrics
        models_history[group_name_to_remove] = {'model': model, 'features': list(current_feature_names), 'accuracy': accuracy, 'auc': auc, 'ks': ks}


    results_df = pd.DataFrame(results)

    # Find the step with the highest accuracy
    best_step_index = results_df['Accuracy'].idxmax()
    best_removed_group = results_df.loc[best_step_index, 'Removed Group']
    best_accuracy = results_df.loc[best_step_index, 'Accuracy']
    best_auc = results_df.loc[best_step_index, 'AUC'] # Get best AUC/KS too
    best_ks = results_df.loc[best_step_index, 'KS']


    # Retrieve the best model details from the history
    best_model_details = models_history[best_removed_group]
    best_model = best_model_details['model']
    best_model_features = best_model_details['features']


    print(f"\nBest model (based on Accuracy) found after removing group: '{best_removed_group}'")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"AUC for Best Model: {best_auc:.4f}")
    print(f"KS for Best Model: {best_ks:.4f}")
    print(f"Number of features in best model: {len(best_model_features)}")


    # Plotting (still plots Accuracy vs Removal Step)
    fig, ax = plt.subplots(figsize=(14, 7))
    num_removed_steps = range(len(results_df))
    ax.plot(num_removed_steps, results_df['Accuracy'], marker='o', linestyle='-')
    ax.scatter(best_step_index, best_accuracy, color='red', s=100, zorder=5, label=f'Best Accuracy ({best_accuracy:.4f})')
    ax.set_xticks(num_removed_steps)
    ax.set_xticklabels(results_df['Removed Group'], rotation=45, ha='right', fontsize=9)
    ax.set_xlabel("Feature Group Removed (Cumulative, Least Important First)")
    ax.set_ylabel("Model Accuracy on Test Set")
    ax.set_title("Sensitivity Analysis: Accuracy vs. Feature Group Removal (Logistic Regression)")
    ax.legend()
    ax.grid(True, axis='y', linestyle='--')
    ax.margins(x=0.02)
    plt.tight_layout()

    # Return test data AND the name of the best removed group
    return results_df, fig, best_model, best_model_features, X_test_full, y_test, best_removed_group

# Function to Parse Classification Report
def parse_classification_report(report_str):
    """
    Parses sklearn classification_report string into a list of lists
    for precision, recall, f1-score, and support (where applicable).
    """
    lines = report_str.strip().split('\n')
    header = [h for h in lines[0].split(' ') if h] # ['precision', 'recall', 'f1-score', 'support']
    data = []
    # Process class lines (0, 1)
    for line in lines[2:-3]: # Skip header, empty line, accuracy, macro, weighted
        row_raw = [r for r in line.split(' ') if r]
        if len(row_raw) == 5: # Class label, precision, recall, f1, support
            data.append(row_raw)
        else:
            print(f"Warning: Skipping unexpected class row format: {line}")

    # Process macro avg line
    macro_line_raw = [l for l in lines[-2].split(' ') if l]
    if len(macro_line_raw) >= 5 and 'avg' in macro_line_raw[1].lower():
        # Keep only metric name, precision, recall, f1. Add empty string for support.
        macro_line = [f"{macro_line_raw[0]} {macro_line_raw[1]}"] + macro_line_raw[2:5] + ['']
        if len(macro_line) == 5:
            data.append(macro_line)
        else:
            print(f"Warning: Could not parse macro avg line correctly: {lines[-2]}")
    else:
        print(f"Warning: Skipping unexpected macro avg format: {lines[-2]}")

    # Process weighted avg line
    weighted_line_raw = [l for l in lines[-1].split(' ') if l]
    if len(weighted_line_raw) >= 5 and 'avg' in weighted_line_raw[1].lower():
        # Keep only metric name, precision, recall, f1. Add empty string for support.
        weighted_line = [f"{weighted_line_raw[0]} {weighted_line_raw[1]}"] + weighted_line_raw[2:5] + ['']
        if len(weighted_line) == 5:
             data.append(weighted_line)
        else:
             print(f"Warning: Could not parse weighted avg line correctly: {lines[-1]}")
    else:
        print(f"Warning: Skipping unexpected weighted avg format: {lines[-1]}")

    return header, data

# Function to Create Table Plot
def plot_classification_report_table(report_str, accuracy, auc, ks, total_support, title, save_path):
    """
    Creates and saves a table plot from a classification report string,
    adding columns for Accuracy, AUC, and KS, populated only in an 'Overall' row.
    The 'Overall' row also includes weighted averages for precision, recall, and f1-score.
    """
    base_header, report_data = parse_classification_report(report_str)

    # Find the weighted average row to extract its metrics
    weighted_avg_precision = ''
    weighted_avg_recall = ''
    weighted_avg_f1 = ''
    for row in report_data:
        if row[0] == 'weighted avg':
            weighted_avg_precision = row[1]
            weighted_avg_recall = row[2]
            weighted_avg_f1 = row[3]
            break # Found it, no need to look further

    num_extra_cols = 3 # Accuracy, AUC, KS

    # Pad existing rows (classes, averages) with empty strings for the new columns
    padded_data = [row + [''] * num_extra_cols for row in report_data]

    # Prepare the 'Overall' row, now including weighted avg values
    overall_row = [
        'Overall',
        weighted_avg_precision, # Use weighted avg precision
        weighted_avg_recall,    # Use weighted avg recall
        weighted_avg_f1,        # Use weighted avg f1-score
        str(total_support),     # Support
        f"{accuracy:.4f}",      # Accuracy
        f"{auc:.4f}",           # AUC
        f"{ks:.4f}"             # KS Stat
    ]

    # Combine standard report data with the overall row
    full_data = padded_data + [overall_row]

    # Define the full header including new columns
    plot_header = ['Metric/Class'] + base_header + ['Accuracy', 'AUC', 'KS Stat']

    # Adjust figsize width to accommodate new columns
    fig, ax = plt.subplots(figsize=(10, 3.5)) # Keep adjusted size
    ax.axis('tight')
    ax.axis('off')

    # Create the table
    the_table = ax.table(cellText=full_data, colLabels=plot_header, loc='center', cellLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1.1, 1.1) # Keep adjusted scale

    # Style the table header and first column
    num_rows = len(full_data)
    num_cols = len(plot_header)
    for (i, j), cell in the_table.get_celld().items():
        if i == 0: # Header row
             cell.set_text_props(weight='bold', color='white')
             cell.set_facecolor('#40466e')
        if j == 0: # First column (Metric/Class)
            cell.set_text_props(weight='bold')
        # Bold the 'Overall' row
        if i == num_rows: # Last row (Overall)
             cell.set_text_props(weight='bold')
        cell.set_edgecolor('lightgrey')

    plt.title(title, y=1.15) # Keep adjusted title position
    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Classification report table saved to: {save_path}")

# Function to Plot Feature Importance as a Table with Grouping (Using Average)
def plot_feature_importance(model, feature_names, title, save_path):
    """
    Creates and saves a table plot of grouped feature importances based on the
    AVERAGE of absolute logistic regression coefficients for each group.

    Args:
        model (LogisticRegression): The trained logistic regression model.
        feature_names (list): List of feature names corresponding to the model's coefficients.
        title (str): The title for the plot.
        save_path (str): The full path to save the plot image.

    Returns:
        matplotlib.figure.Figure: The figure object of the plot, or None if failed.
    """
    if not hasattr(model, 'coef_'):
        print(f"Error: Model for '{title}' does not have coefficients. Cannot plot importance.")
        return None

    coefficients = model.coef_[0]
    if len(coefficients) != len(feature_names):
         print(f"Error: Mismatch between number of coefficients ({len(coefficients)}) and feature names ({len(feature_names)}) for '{title}'.")
         return None

    feature_importance_map = dict(zip(feature_names, np.abs(coefficients)))
    feature_groups = get_feature_groups(feature_names)

    group_importances = {}
    for group_name, group_cols in feature_groups.items():
        num_cols = len(group_cols)
        if num_cols > 0:
            total_importance = sum(feature_importance_map.get(col, 0) for col in group_cols)
            average_importance = total_importance / num_cols
        else:
            average_importance = 0
        group_importances[group_name] = average_importance

    importance_df = pd.DataFrame(list(group_importances.items()), columns=['Feature Group', 'Average Importance'])
    importance_df = importance_df.sort_values(by='Average Importance', ascending=False).reset_index(drop=True)
    importance_df['Average Importance'] = importance_df['Average Importance'].map('{:,.4f}'.format)

    num_rows = len(importance_df)
    fig_height = max(3, num_rows * 0.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.axis('tight')
    ax.axis('off')

    table_data = importance_df.values.tolist()
    col_labels = importance_df.columns.tolist()

    the_table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1.1, 1.1)

    for (i, j), cell in the_table.get_celld().items():
        if i == 0: # Header row
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        if j == 0: # First column (Feature Group)
             cell.set_text_props(ha='left')
        else: # Second column (Importance)
             cell.set_text_props(ha='right')
        cell.set_edgecolor('lightgrey')

    plt.title(title, y=1.05, fontsize=12)
    plt.tight_layout(pad=1.5)
    fig = plt.gcf()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Feature importance table (grouped avg) saved to: {save_path}")
    return fig

# Function to Plot Individual Feature Importance as a Table (All Features)
def plot_individual_feature_importance(model, feature_names, title, save_path):
    """
    Creates and saves a table plot of all individual feature importances
    based on the absolute logistic regression coefficients, sorted descending.

    Args:
        model (LogisticRegression): The trained logistic regression model.
        feature_names (list): List of feature names corresponding to the model's coefficients.
        title (str): The title for the plot.
        save_path (str): The full path to save the plot image.

    Returns:
        matplotlib.figure.Figure: The figure object of the plot, or None if failed.
    """
    if not hasattr(model, 'coef_'):
        print(f"Error: Model for '{title}' does not have coefficients. Cannot plot importance.")
        return None

    coefficients = model.coef_[0]
    if len(coefficients) != len(feature_names):
         print(f"Error: Mismatch between number of coefficients ({len(coefficients)}) and feature names ({len(feature_names)}) for '{title}'.")
         return None

    # Map feature names to their absolute coefficient values
    importance_df = pd.DataFrame({'Feature': feature_names, 'Absolute Coefficient': np.abs(coefficients)})
    importance_df = importance_df.sort_values(by='Absolute Coefficient', ascending=False).reset_index(drop=True)

    # No longer limiting to top N features
    plot_title = title # Use the title directly

    # Format the importance values for display
    importance_df['Absolute Coefficient'] = importance_df['Absolute Coefficient'].map('{:,.4f}'.format)

    # --- Create Table Plot ---
    num_rows = len(importance_df)
    # Increase height multiplier to accommodate potentially more rows
    fig_height = max(4, num_rows * 0.35) # Adjusted height calculation
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.axis('tight')
    ax.axis('off')

    table_data = importance_df.values.tolist()
    col_labels = importance_df.columns.tolist()

    the_table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9) # Slightly smaller font if there are many rows
    the_table.scale(1.1, 1.1)

    for (i, j), cell in the_table.get_celld().items():
        if i == 0: # Header row
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        if j == 0: # First column (Feature)
             cell.set_text_props(ha='left')
        else: # Second column (Coefficient)
             cell.set_text_props(ha='right')
        cell.set_edgecolor('lightgrey')

    plt.title(plot_title, y=1.02, fontsize=12) # Adjust title position slightly
    plt.tight_layout(pad=1.5)
    fig = plt.gcf()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Feature importance table (individual - all) saved to: {save_path}") # Clarified print
    return fig

# --- Directory Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
graficos_dir = os.path.join(current_dir, 'graficos', 'modelo_regresion')
os.makedirs(graficos_dir, exist_ok=True)

# --- DataFrame Loading ---
df = pd.read_csv('punto 2/diabetes_binary_health_indicators_BRFSS2015_transformed.csv')
print(f"Loaded DataFrame shape: {df.shape}")
# df = df.head(1000) # Uncomment for quick testing
# print(f"Using subset DataFrame shape: {df.shape}")

# --- Main Execution Logic ---

# --- Run Model 1 (All Features) ---
print("\n--- Running Model 1 (All Features) ---")
model, accuracy, auc, ks, cm, report, y_test_model1 = run_logistic_regression(df, 'Diabetes_binary')

# --- Run Model 2 (Selected Features) ---
print("\n--- Running Model 2 (Selected Features) ---")
features_model2 = [
    'HighBP', 'HighChol', 'DiffWalk',
    # BMI Categories
    'BMI_Category_Normal', 'BMI_Category_Overweight', 'BMI_Category_Obese_I',
    'BMI_Category_Obese_II', 'BMI_Category_Obese_III',
    # Age Categories
    'Age_2', 'Age_3', 'Age_4', 'Age_5', 'Age_6', 'Age_7', 'Age_8', 'Age_9',
    'Age_10', 'Age_11', 'Age_12', 'Age_13',
    # GenHlth Categories
    'GenHlth_2', 'GenHlth_3', 'GenHlth_4', 'GenHlth_5'
]
columns_for_df2 = ['Diabetes_binary'] + features_model2
df2 = df[columns_for_df2].copy()
model2, accuracy2, auc2, ks2, cm2, report2, _ = run_logistic_regression(df2, 'Diabetes_binary')

# --- Sensitivity Analysis ---
print("\n--- Sensitivity Analysis for Model (All Features) ---")
results_df, fig_sens, best_model_sens, best_features_sens, X_test_data_sens, y_test_data_sens, best_removed_group_sens = sensitivity_analysis_logistic_regression(df, 'Diabetes_binary', random_state=42)

# Display and save sensitivity analysis results
print("\nSensitivity Analysis Results:")
print(results_df)
plot_path_sens = os.path.join(graficos_dir, 'sensitivity_analysis_model.png')
if fig_sens: # Check if figure was created
    fig_sens.savefig(plot_path_sens)
    print(f"\nSensitivity analysis plot saved to: {plot_path_sens}")
else:
    print("\nSensitivity analysis plot was not generated (likely due to model coefficient issues).")


# --- Metrics and Reports for Best Sensitivity Model ---
if best_model_sens: # Check if a best model was found
    best_step_index = results_df['Accuracy'].idxmax()
    best_accuracy_sens = results_df.loc[best_step_index, 'Accuracy']
    best_auc_sens = results_df.loc[best_step_index, 'AUC']
    best_ks_sens = results_df.loc[best_step_index, 'KS']

    print(f"\nStored the best model from sensitivity analysis (best_model_sens) with Accuracy: {best_accuracy_sens:.4f}")
    print(f"Features used in the best model ({len(best_features_sens)}): {best_features_sens}")

    # Calculate CM and Report String for the best model
    X_test_best_features_sens = X_test_data_sens[best_features_sens]
    y_pred_best_sens = best_model_sens.predict(X_test_best_features_sens)
    cm_best_sens = confusion_matrix(y_test_data_sens, y_pred_best_sens)
    report_best_sens = classification_report(y_test_data_sens, y_pred_best_sens)

    # Print Metrics
    print("\nMetrics for Best Model from Sensitivity Analysis (best_model_sens) on Test Set:")
    print(f"Accuracy: {best_accuracy_sens:.4f}")
    print(f"AUC: {best_auc_sens:.4f}")
    print(f"KS Statistic: {best_ks_sens:.4f}")
    print("\nConfusion Matrix:")
    print(cm_best_sens)
    print("\nClassification Report:")
    print(report_best_sens)
else:
    print("\nBest model from sensitivity analysis could not be determined.")
    # Assign default/None values to avoid errors later if sensitivity analysis failed
    best_accuracy_sens = None
    best_auc_sens = None
    best_ks_sens = None
    report_best_sens = "N/A"
    y_test_data_sens = [] # Empty list for total support calculation


# --- Generate Classification Report Table Plots ---

# Model 1 (Full Features)
total_support_model1 = len(y_test_model1)
report_table_path_model1 = os.path.join(graficos_dir, 'classification_report_model1.png')
plot_classification_report_table(
    report, # From initial full model run
    accuracy,
    auc,
    ks,
    total_support_model1,
    "Classification Report: Model 1 (All Features)",
    report_table_path_model1
)

# Best Model from Sensitivity Analysis
if best_model_sens: # Only plot if the best model exists
    total_support_best_sens = len(y_test_data_sens)
    report_table_path_best_sens = os.path.join(graficos_dir, 'classification_report_best_model_sens.png')
    plot_classification_report_table(
        report_best_sens,
        best_accuracy_sens,
        best_auc_sens,
        best_ks_sens,
        total_support_best_sens,
        f"Classification Report: Best Sensitivity Model",
        report_table_path_best_sens
    )
else:
    print("\nSkipping classification report table plot for best sensitivity model as it wasn't generated.")


# --- Generate Feature Importance Plots (Grouped Average AND Individual All) ---

# Get feature names for the full model
full_model_feature_names = df.drop('Diabetes_binary', axis=1).columns.tolist()

# --- Model 1 Plots ---
# Grouped Average Importance
importance_plot_path_model1_avg = os.path.join(graficos_dir, 'feature_importance_model1_avg_table.png')
plot_feature_importance(
    model,
    full_model_feature_names,
    "Avg Feature Importance: Model 1 (Grouped)",
    importance_plot_path_model1_avg
)
# Individual Importance (All)
importance_plot_path_model1_ind = os.path.join(graficos_dir, 'feature_importance_model1_individual_table.png')
plot_individual_feature_importance( # No top_n argument needed
    model,
    full_model_feature_names,
    "Individual Feature Importance: Model 1",
    importance_plot_path_model1_ind
)

# --- Best Model Plots ---
if best_model_sens:
    # Grouped Average Importance
    importance_plot_path_best_avg = os.path.join(graficos_dir, 'feature_importance_best_model_avg_table.png')
    plot_feature_importance(
        best_model_sens,
        best_features_sens,
        f"Avg Feature Importance: Best Model (Grouped)",
        importance_plot_path_best_avg
    )
    # Individual Importance (All)
    importance_plot_path_best_ind = os.path.join(graficos_dir, 'feature_importance_best_model_individual_table.png')
    plot_individual_feature_importance( # No top_n argument needed
        best_model_sens,
        best_features_sens,
        f"Individual Feature Importance: Best Model",
        importance_plot_path_best_ind
    )
else:
    print("\nSkipping feature importance plots for best sensitivity model as it wasn't generated.")


# --- Final Plot Display and Cleanup ---
print("\nDisplaying generated plots (if any)...")
plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figures...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')