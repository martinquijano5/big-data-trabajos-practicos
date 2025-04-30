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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
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

def run_logistic_regression(df, target_column, test_size=0.3, random_state=42):
    """
    Runs a logistic regression model on the provided DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing features and target variable.
        target_column (str): The name of the target variable column.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before splitting.

    Returns:
        tuple: Contains the trained model, accuracy score, confusion matrix,
               and classification report.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


    # Initialize and train the Logistic Regression model
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\nLogistic Regression Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    return model, accuracy, cm, report


current_dir = os.path.dirname(os.path.abspath(__file__))
graficos_dir = os.path.join(current_dir, 'graficos', 'modelo_regresion')
os.makedirs(graficos_dir, exist_ok=True)


df = pd.read_csv('punto 2/diabetes_binary_health_indicators_BRFSS2015_transformed.csv')

print("--- Running Model 1 (All Features) ---")
model, accuracy, cm, report = run_logistic_regression(df, 'Diabetes_binary')

print("\n--- Running Model 2 (Selected Features) ---")

# Define the columns for the second model based on your request and the df.info() output
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

# Include the target variable
columns_for_df2 = ['Diabetes_binary'] + features_model2

# Create a new DataFrame with only the selected columns
df2 = df[columns_for_df2].copy()

# Run the logistic regression on the subset of features
model2, accuracy2, cm2, report2 = run_logistic_regression(df2, 'Diabetes_binary')

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
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(random_state=random_state, max_iter=1000, solver='liblinear') 
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Return the scaler as well, it's needed if we want to use the model later
    return model, accuracy, scaler 

# Main sensitivity analysis function
def sensitivity_analysis_logistic_regression(df, target_column, random_state=42):
    """
    Performs sensitivity analysis by iteratively removing feature groups 
    based on logistic regression coefficient magnitudes. Identifies and 
    returns the model with the highest accuracy during the process, along with the test data.

    Args:
        df (pd.DataFrame): DataFrame containing features and target variable.
        target_column (str): The name of the target variable column.
        random_state (int): Controls the shuffling and model training.

    Returns:
        tuple: (results_df, fig, best_model, best_model_features, best_scaler, X_test_data, y_test_data)
            results_df (pd.DataFrame): DataFrame tracking removed groups and accuracy.
            fig (matplotlib.figure.Figure): Figure object for the accuracy plot.
            best_model (LogisticRegression): The trained model instance with the highest accuracy.
            best_model_features (list): List of feature names used by the best model.
            best_scaler (StandardScaler): The scaler fitted on the training data for the best model.
            X_test_data (pd.DataFrame): The feature data from the test split.
            y_test_data (pd.Series): The target data from the test split.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    feature_names = X.columns.tolist()

    # 1. Initial Data Split (consistent across iterations)
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )

    # 2. Train Initial Model (baseline)
    print("Training initial model for baseline and feature importance...")
    initial_model, initial_accuracy, initial_scaler = run_logistic_regression_minimal(
        X_train_full, y_train, X_test_full, y_test, random_state=random_state
    )
    print(f"Initial Accuracy (All Features): {initial_accuracy:.4f}")

    # Store models and their features at each step
    models_history = {'None': {'model': initial_model, 'features': list(feature_names), 'accuracy': initial_accuracy, 'scaler': initial_scaler}}
    results = {'Removed Group': ['None'], 'Accuracy': [initial_accuracy]}
    
    if not hasattr(initial_model, 'coef_'):
         print("Error: Could not retrieve coefficients from the initial model.")
         # Return None for the test data as well
         return pd.DataFrame(results), None, None, [], None, None, None 

    # 3. Group Features
    feature_groups = get_feature_groups(feature_names)
    print(f"Identified {len(feature_groups)} feature groups.")

    # 4. Calculate Group Importance from Initial Model
    # Use coefficients from the positive class prediction (index 0 for binary classification)
    initial_coeffs = initial_model.coef_[0] 
    feature_importance_map = dict(zip(feature_names, np.abs(initial_coeffs)))
    
    group_importances = {}
    for group_name, group_cols in feature_groups.items():
        # Sum of absolute coefficients for features in the group
        importance = sum(feature_importance_map.get(col, 0) for col in group_cols)
        group_importances[group_name] = importance

    # 5. Sort Groups by Importance (Ascending - least important first)
    sorted_groups = sorted(group_importances.items(), key=lambda item: item[1])
    
    # 6. Iterative Removal and Retraining
    current_feature_names = list(feature_names) 

    print("\nStarting iterative feature group removal (least important first)...")
    
    # Iterate through sorted groups, removing one group at a time
    # We iterate up to len(sorted_groups) - 1 to avoid removing the last group
    for i in range(len(sorted_groups) - 1): 
        group_name_to_remove, group_importance = sorted_groups[i]
        cols_to_remove = feature_groups[group_name_to_remove]
        
        print(f"Step {i+1}: Removing group '{group_name_to_remove}' (Importance: {group_importance:.4f}, Features: {len(cols_to_remove)})")

        # Update the list of current features by removing the group's columns
        current_feature_names = [f for f in current_feature_names if f not in cols_to_remove]

        if not current_feature_names:
            print("  No features left after removal. Stopping analysis.")
            break

        # Select remaining columns from the original train/test sets
        remaining_X_train = X_train_full[current_feature_names]
        remaining_X_test = X_test_full[current_feature_names]
        
        # Re-train model and get scaler
        model, accuracy, scaler = run_logistic_regression_minimal(
            remaining_X_train, y_train, remaining_X_test, y_test, random_state=random_state
        )
        
        print(f"  Accuracy after removal: {accuracy:.4f}")

        # Store result
        results['Removed Group'].append(group_name_to_remove)
        results['Accuracy'].append(accuracy)
        # Store the model, its features, and scaler for this step
        models_history[group_name_to_remove] = {'model': model, 'features': list(current_feature_names), 'accuracy': accuracy, 'scaler': scaler}


    results_df = pd.DataFrame(results)

    # Find the step with the highest accuracy
    best_step_index = results_df['Accuracy'].idxmax()
    best_removed_group = results_df.loc[best_step_index, 'Removed Group']
    best_accuracy = results_df.loc[best_step_index, 'Accuracy']

    # Retrieve the best model details from the history
    # The key in models_history is the group *removed* to get this result
    # So, if best_removed_group is 'None', it's the initial model. Otherwise it's the model trained *after* removing that group.
    best_model_details = models_history[best_removed_group]
    best_model = best_model_details['model']
    best_model_features = best_model_details['features']
    best_scaler = best_model_details['scaler']
    
    print(f"\nBest model found after removing group: '{best_removed_group}'")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Number of features in best model: {len(best_model_features)}")


    # Plotting
    fig, ax = plt.subplots(figsize=(14, 7)) 
    num_removed_steps = range(len(results_df)) 
    ax.plot(num_removed_steps, results_df['Accuracy'], marker='o', linestyle='-')
    ax.scatter(best_step_index, best_accuracy, color='red', s=100, zorder=5, label=f'Best Accuracy ({best_accuracy:.4f})') # Highlight best point
    ax.set_xticks(num_removed_steps)
    ax.set_xticklabels(results_df['Removed Group'], rotation=45, ha='right', fontsize=9) 
    ax.set_xlabel("Feature Group Removed (Cumulative, Least Important First)")
    ax.set_ylabel("Model Accuracy on Test Set")
    ax.set_title("Sensitivity Analysis: Accuracy vs. Feature Group Removal (Logistic Regression)")
    ax.legend()
    ax.grid(True, axis='y', linestyle='--')
    ax.margins(x=0.02) 
    plt.tight_layout() 

    # Return the test data as well
    return results_df, fig, best_model, best_model_features, best_scaler, X_test_full, y_test

# --- Updated usage ---
print("\n--- Sensitivity Analysis for Model (All Features) ---") # Renamed title slightly
# Capture the returned best model, its features, scaler, and test data
results_df, fig, best_model_m, best_features_m, best_scaler_m, X_test_data, y_test_data = sensitivity_analysis_logistic_regression(
    df, 'Diabetes_binary', random_state=42
)

# Display the results table
print("\nSensitivity Analysis Results:")
print(results_df)

# Save the plot
plot_path = os.path.join(graficos_dir, 'sensitivity_analysis_model.png')
fig.savefig(plot_path)
print(f"\nSensitivity analysis plot saved to: {plot_path}")


best_accuracy = results_df['Accuracy'].max()
print(f"\nStored the best model (best_model_m) with accuracy: {best_accuracy:.4f}")
print(f"Features used in the best model ({len(best_features_m)}): {best_features_m}")

# --- Calculate and Print Confusion Matrix for the Best Model ---
# 1. Select the features used by the best model from the test set
X_test_best_features = X_test_data[best_features_m]


# 2. Make predictions on the scaled test data
y_pred_best = best_model_m.predict(X_test_best_features)

# 3. Calculate the confusion matrix
cm_best = confusion_matrix(y_test_data, y_pred_best)

# 4. Print the confusion matrix
print("\nConfusion Matrix for Best Model (best_model_m) on Test Set:")
print(cm_best)
print("\nClassification Report for Best Model:")
print(classification_report(y_test_data, y_pred_best))



# Show all plots
plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figures...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')