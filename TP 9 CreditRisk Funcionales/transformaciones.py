import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import os

def apply_transformations_complete(df):
    df_transformed = df.copy()
    
    transformations = {
        1: {"new_name": "status-cuenta", "mapping": {"A14": 2, "A11": 1, "A12": 3, "A13": 4}},
        
        2: {
            "new_name": "duracion-meses",
            "transform_func": lambda x: (
                1 if pd.notna(x) and x <= 10 else
                2 if pd.notna(x) and x <= 20 else
                3 if pd.notna(x) and x <= 30 else
                4 if pd.notna(x) and x <= 40 else 5
            )
        },
        
        3: {"new_name": "credit-history", "mapping": {"A30": 1, "A31": 2, "A32": 3, "A33": 4, "A34": 5}},
        
        4: {
            "new_name": "credit-purpose", 
            "mapping": {"A49": 1, "A48": 2, "A47": 3, "A46": 4, "A45": 5, 
                        "A44": 6, "A43": 7, "A42": 8, "A41": 9, "A40": 10, "A410": 11}
        },
        
        5: {"new_name": "credit-amount", "no_transform": True},
        
        6: {"new_name": "saving-account-amount", "mapping": {"A65": 1, "A61": 2, "A62": 3, "A63": 2, "A64": 1}},
        
        7: {"new_name": "antiguedad-trabajo", "mapping": {"A75": 1, "A74": 2, "A73": 3, "A72": 4, "A71": 5}},
        
        8: {"new_name": "tasa-interes", "no_transform": True},
        
        9: {"new_name": "estado-civil", "mapping": {"A91": 1, "A92": 2, "A93": 3, "A94": 4, "A95": 5}},
        
        10: {"new_name": "garante", "mapping": {"A101": 3, "A102": 2, "A103": 1}},
        
        11: {"new_name": "11", "no_transform": True},
        
        12: {"new_name": "propiedades", "mapping": {"A124": 4, "A123": 3, "A122": 2, "A121": 1}},
        
        13: {
            "new_name": "edad",
            "transform_func": lambda x: 1 if pd.notna(x) and x < 30 else 0
        },
        
        14: {"new_name": "14", "mapping": {"A141": 141, "A142": 142, "A143": 143}},
        
        15: {"new_name": "alojamiento", "mapping": {"A153": 1, "A151": 2, "A152": 3}},
        
        16: {"new_name": "cantidad-creditos", "no_transform": True},
        
        17: {"new_name": "trabajo", "mapping": {"A171": 4, "A172": 3, "A173": 2, "A174": 1}},
        
        18: {"new_name": "cantidad-manutencion", "no_transform": True},
        
        19: {"new_name": "telefono", "mapping": {"A191": 1, "A192": 0}},
        
        20: {"new_name": "trabajo-domestico", "mapping": {"A201": 0, "A202": 1}},
    }
    
    for col_num, transform_info in transformations.items():
        if col_num in df_transformed.columns:
            new_name = transform_info["new_name"]
            
            if "mapping" in transform_info:
                df_transformed[col_num] = df_transformed[col_num].map(transform_info["mapping"])
            elif "transform_func" in transform_info:
                if new_name in ["duracion-meses", "edad"]:
                    df_transformed[col_num] = pd.to_numeric(df_transformed[col_num], errors='coerce')
                df_transformed[col_num] = df_transformed[col_num].apply(transform_info["transform_func"])
            
            df_transformed = df_transformed.rename(columns={col_num: new_name})
    
    if "Rechazo" in df_transformed.columns:
        df_transformed["Rechazo"] = df_transformed["Rechazo"].map({1: 0, 2: 1})
    
    columns_to_drop_final = [col for col in ["11", "14"] if col in df_transformed.columns]
    if columns_to_drop_final:
        df_transformed = df_transformed.drop(columns=columns_to_drop_final)
    
    return df_transformed

# Load the Excel file
print("Loading the dataset...")
try:
    df = pd.read_excel("Base_Clientes Alemanes.xlsx")
except FileNotFoundError:
    print("Error: 'Base_Clientes Alemanes.xlsx' not found. Make sure the file is in the correct directory.")
    exit()

# Apply the transformations
print("Applying transformations...")
df_transformed_complete = apply_transformations_complete(df.copy())

# Save the transformed DataFrame to a new Excel file
output_dir_transformed = os.path.dirname("Base_Clientes Alemanes Transformed.xlsx")
if output_dir_transformed and not os.path.exists(output_dir_transformed):
    os.makedirs(output_dir_transformed, exist_ok=True)
output_file_name_transformed = "Base_Clientes Alemanes Transformed.xlsx"
df_transformed_complete.to_excel(output_file_name_transformed, index=False)
print(f"Transformed data successfully saved to '{output_file_name_transformed}'")

print("\nStarting one-hot encoding and final processing...")

# Separate target variable (assuming 'Rechazo' is present and correctly mapped)
if 'Rechazo' not in df_transformed_complete.columns:
    print("Error: 'Rechazo' column not found in transformed data. Check transformations.")
    exit()

y_final = df_transformed_complete['Rechazo']
X_final = df_transformed_complete.drop('Rechazo', axis=1)

# Identify which columns should be treated as categorical based on their unique values
categorical_columns_final = []
numerical_columns_final = []

for col in X_final.columns:
    n_unique = X_final[col].nunique(dropna=False)
    if n_unique <= 10 and X_final[col].dtype != 'float64' and X_final[col].dtype != 'int64' :
         categorical_columns_final.append(col)
    elif X_final[col].dtype == 'object':
        categorical_columns_final.append(col)
    
    known_continuous_cols = ['credit-amount', 'tasa-interes', 'cantidad-creditos', 'cantidad-manutencion']
    
    if col not in known_continuous_cols and X_final[col].nunique(dropna=False) <= 10:
        categorical_columns_final.append(col)
    else:
        numerical_columns_final.append(col)

print("\nColumns identified as categorical for one-hot encoding:", categorical_columns_final)
print("Columns identified as numerical:", numerical_columns_final)

# Apply one-hot encoding to identified categorical variables
if categorical_columns_final:
    print("\nApplying one-hot encoding to categorical variables...")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    existing_categorical_columns = [col for col in categorical_columns_final if col in X_final.columns]
    if not existing_categorical_columns:
        print("No valid categorical columns found for encoding. Skipping OHE.")
        categorical_df_final = pd.DataFrame(index=X_final.index)
    else:
        categorical_data_final = encoder.fit_transform(X_final[existing_categorical_columns])
        
        feature_names_final = []
        for i, col_name in enumerate(existing_categorical_columns):
            categories = encoder.categories_[i]
            for cat in categories:
                feature_names_final.append(f"{col_name}_{cat}")
        
        categorical_df_final = pd.DataFrame(categorical_data_final, columns=feature_names_final, index=X_final.index)
else:
    print("\nNo categorical columns to one-hot encode.")
    categorical_df_final = pd.DataFrame(index=X_final.index)

existing_numerical_columns = [col for col in numerical_columns_final if col in X_final.columns]

final_df_processed = pd.concat([
    categorical_df_final,
    X_final[existing_numerical_columns].reset_index(drop=True),
    y_final.reset_index(drop=True)
], axis=1)

print("\nFinal combined dataset shape (after OHE):", final_df_processed.shape)
print("\nFirst few rows of the final OHE DataFrame:")
print(final_df_processed.head())

# Save the final combined DataFrame
output_dir_final_csv = os.path.dirname("clientes_final.csv")
if output_dir_final_csv and not os.path.exists(output_dir_final_csv):
    os.makedirs(output_dir_final_csv, exist_ok=True)
final_df_processed.to_csv("clientes_final.csv", index=False)
print("\nFinal combined DataFrame (clientes_final.csv) has been saved.")

# Prepare data for model training (Random Forest part, kept from original transformaciones.py)
print("\nPreparing data for Random Forest model training (using pre-OHE data)...")
X_rf = df_transformed_complete.drop('Rechazo', axis=1)
y_rf = df_transformed_complete['Rechazo']

# Split the data into training and testing sets
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42, stratify=y_rf)

# Define class weights to handle class balancing
class_weights_rf = {0: 1, 1: 5}

# Initialize and train the Random Forest model with class weights
print("\nTraining Random Forest model with class weights...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight=class_weights_rf,
    random_state=42
)

# Fit the model
rf_model.fit(X_train_rf, y_train_rf)

# Make predictions
y_pred_rf = rf_model.predict(X_test_rf)

# Calculate metrics
accuracy_rf = accuracy_score(y_test_rf, y_pred_rf)
precision_rf = precision_score(y_test_rf, y_pred_rf)
recall_rf = recall_score(y_test_rf, y_pred_rf)
f1_rf = f1_score(y_test_rf, y_pred_rf)

# Print metrics
print("\nRandom Forest Model Performance Metrics:")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1 Score: {f1_rf:.4f}")

# Create confusion matrix
cm_rf = confusion_matrix(y_test_rf, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plot_output_dir = "graficos/transformaciones"
if not os.path.exists(plot_output_dir):
    os.makedirs(plot_output_dir, exist_ok=True)
plt.savefig(os.path.join(plot_output_dir, "confusion_matrix_rf.png"))
plt.close()

# Feature importance plot
feature_importance_rf = pd.DataFrame({
    'feature': X_rf.columns,
    'importance': rf_model.feature_importances_
})
feature_importance_rf = feature_importance_rf.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_rf.head(15))
plt.title('Top 15 Most Important Features (Random Forest)')
plt.tight_layout()
plt.savefig(os.path.join(plot_output_dir, "feature_importance_rf.png"))
plt.close()

print("\nRandom Forest model training and evaluation completed. Plots have been saved.")