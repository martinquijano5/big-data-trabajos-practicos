import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from scipy.stats import mode
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Set style for better visualizations
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the dataset
df = pd.read_csv('TP 2 Churn Iran customer/Iran Customer Churn.csv')

# Basic data exploration
print("Head of the dataset:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nDescriptive statistics:")
print(df.describe())


# Check class distribution (churn vs non-churn)
print("\nChurn distribution:")
churn_counts = df['Churn'].value_counts()
print(churn_counts)
print(f"Churn rate: {churn_counts[1] / len(df) * 100:.2f}%")


# Visualizations
# 1. Churn Distribution - Pie Chart
plt.figure(figsize=(10, 6))
plt.pie(churn_counts, labels=['Non-Churn', 'Churn'], autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
plt.title('Churn Distribution')

# 2. Correlation Heatmap
plt.figure(figsize=(14, 10))
correlation = df.corr()
mask = np.triu(correlation)
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5, mask=mask)
plt.title('Correlation Heatmap')
plt.tight_layout()

# 3. Numerical Features Distribution by Churn - Split into multiple figures with 4 variables each
numerical_features = [
    'Call  Failure', 'Seconds of Use', 'Frequency of use', 'Frequency of SMS', 
    'Distinct Called Numbers', 'Subscription  Length', 'Charge  Amount', 'Age Group',
    'Customer Value'  
]

# Split features into groups of 4
feature_groups = [numerical_features[i:i+4] for i in range(0, len(numerical_features), 4)]

# Create a separate figure for each group
for group_idx, feature_group in enumerate(feature_groups):
    # For the last group with fewer than 4 features, adjust the layout
    if len(feature_group) < 4:
        # If only 1 feature, create a single plot
        if len(feature_group) == 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='Churn', y=feature_group[0], data=df, ax=ax)
            ax.set_title(f'{feature_group[0]} by Churn Status')
            ax.set_xlabel('Churn (1=Yes, 0=No)')
        else:
            # For 2 or 3 features, create a 1-row subplot
            fig, axes = plt.subplots(1, len(feature_group), figsize=(15, 6))
            if len(feature_group) == 1:  # If only one subplot, axes won't be an array
                axes = [axes]
            
            for i, feature in enumerate(feature_group):
                sns.boxplot(x='Churn', y=feature, data=df, ax=axes[i])
                axes[i].set_title(f'{feature} by Churn Status')
                axes[i].set_xlabel('Churn (1=Yes, 0=No)')
    else:
        # For groups with 4 features, create a 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(feature_group):
            sns.boxplot(x='Churn', y=feature, data=df, ax=axes[i])
            axes[i].set_title(f'{feature} by Churn Status')
            axes[i].set_xlabel('Churn (1=Yes, 0=No)')
    
    plt.tight_layout()

# 4. Categorical Features Analysis
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
categorical_features = ['Complains', 'Tariff Plan', 'Status']

for i, feature in enumerate(categorical_features):
    # Create a crosstab
    ct = pd.crosstab(df[feature], df['Churn'])
    # Calculate percentages
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    
    # Plot
    ct_pct.plot(kind='bar', stacked=True, ax=axes[i], color=['#66b3ff', '#ff9999'])
    axes[i].set_title(f'Churn Rate by {feature}')
    axes[i].set_ylabel('Percentage')
    axes[i].set_xlabel(feature)
    
    # Add percentage labels
    for c in axes[i].containers:
        axes[i].bar_label(c, label_type='center', fmt='%.1f%%')



# 5. Predictive Modeling - KNN for Churn Prediction
print("\n--- KNN Classification Analysis for Churn Prediction ---")

# Prepare data for modeling
X = df.drop(['Churn', 'Age'], axis=1)  # All features except Churn and Age
y = df['Churn']  # Target variable

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find the optimal k value using cross-validation
k_range = range(1, 51)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y, cv=10, scoring='accuracy')  # 10-fold cross-validation
    cv_scores.append(scores.mean())

# Plot accuracy vs k value
plt.figure(figsize=(10, 6))
plt.plot(k_range, cv_scores)
plt.xlabel('Value of K')
plt.ylabel('Cross-Validated Accuracy')
plt.title('KNN Accuracy for Different Values of K')
plt.grid(True)
plt.tight_layout()

# Find the optimal k (lowest k with highest accuracy)
best_k = 0
best_accuracy = 0
for i, score in enumerate(cv_scores):
    if score > best_accuracy:
        best_accuracy = score
        best_k = k_range[i]
    elif score == best_accuracy and k_range[i] < best_k:
        best_k = k_range[i]

print(f"\nBest k value from cross-validation: {best_k} with accuracy: {best_accuracy:.4f}")

# Train the final model with the optimal k
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_scaled, y)

# Get predictions using cross-validation to evaluate model performance
y_pred = cross_val_predict(best_knn, X_scaled, y, cv=10)
final_accuracy = accuracy_score(y, y_pred)

print(f"\nFinal model accuracy: {final_accuracy:.4f}")
print("\nClassification report:")
print(classification_report(y, y_pred))

# Confusion matrix
print("\nConfusion matrix:")
conf_matrix = confusion_matrix(y, y_pred)
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Churn (0)', 'Churn (1)'],
            yticklabels=['Non-Churn (0)', 'Churn (1)'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'KNN Confusion Matrix (k={best_k})')
plt.tight_layout()


# 6. Logistic Regression for Churn Prediction
print("\n--- Logistic Regression Analysis for Churn Prediction ---")

# Define hyperparameters to tune
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
penalties = ['l1', 'l2', 'elasticnet', None]
solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

# Not all combinations of penalty and solver are valid, so create valid combinations
param_grid = []
for penalty in penalties:
    if penalty == 'l1':
        param_grid.append({'C': C_values, 'penalty': ['l1'], 'solver': ['liblinear', 'saga']})
    elif penalty == 'l2':
        param_grid.append({'C': C_values, 'penalty': ['l2'], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']})
    elif penalty == 'elasticnet':
        param_grid.append({'C': C_values, 'penalty': ['elasticnet'], 'solver': ['saga']})
    elif penalty is None:
        param_grid.append({'C': [1.0], 'penalty': [None], 'solver': ['newton-cg', 'lbfgs', 'sag', 'saga']})

# Use GridSearchCV to find the best hyperparameters
print("\nPerforming grid search for logistic regression hyperparameters...")
grid_search = GridSearchCV(
    estimator=LogisticRegression(max_iter=2000, random_state=42),
    param_grid=param_grid,
    cv=10,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Perform the grid search
grid_search.fit(X_scaled, y)

# Get the best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"\nBest hyperparameters: {best_params}")
print(f"Best cross-validation accuracy: {best_score:.4f}")

# Train the final model with the best parameters
best_log_reg = LogisticRegression(**best_params, random_state=42, max_iter=2000)
best_log_reg.fit(X_scaled, y)

# Get predictions using cross-validation to evaluate model performance
log_y_pred = cross_val_predict(best_log_reg, X_scaled, y, cv=10)
log_final_accuracy = accuracy_score(y, log_y_pred)

print(f"\nFinal logistic regression model accuracy: {log_final_accuracy:.4f}")
print("\nClassification report:")
print(classification_report(y, log_y_pred))

# Confusion matrix
print("\nConfusion matrix:")
log_conf_matrix = confusion_matrix(y, log_y_pred)
print(log_conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(log_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Churn (0)', 'Churn (1)'],
            yticklabels=['Non-Churn (0)', 'Churn (1)'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression Confusion Matrix')
plt.tight_layout()

# Analyze feature importance through coefficients
if best_params['penalty'] != 'l1' or best_params['C'] > 0.01:  # Only if coefficients are significant
    plt.figure(figsize=(12, 8))
    feature_names = X.columns
    
    # For binary classification, we only have one set of coefficients
    # But we'll create a DataFrame with a single row to match the heatmap style
    coef_df = pd.DataFrame(
        [best_log_reg.coef_[0]], 
        columns=feature_names,
        index=['Churn (1)']
    )
    
    # Plot the coefficients as a heatmap
    sns.heatmap(coef_df, annot=True, cmap='coolwarm', fmt='.3f')
    plt.title('Coeficientes de regresion logistica por clase (modelo final)')
    plt.tight_layout()


print("\n--- Random Forest Analysis for Churn Prediction ---")

# Define the parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': [None] + list(range(5, 30, 5)),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

# Initialize Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Use RandomizedSearchCV to find good hyperparameters
print("\nPerforming randomized search for Random Forest hyperparameters...")
rf_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,  # Number of parameter settings sampled
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Perform the randomized search
rf_search.fit(X_scaled, y)

# Get the best parameters and score
rf_best_params = rf_search.best_params_
rf_best_score = rf_search.best_score_

print(f"\nBest hyperparameters: {rf_best_params}")
print(f"Best cross-validation accuracy: {rf_best_score:.4f}")

# Train the final model with the best parameters
best_rf = RandomForestClassifier(**rf_best_params, random_state=42)
best_rf.fit(X_scaled, y)

# Get predictions using cross-validation to evaluate model performance
rf_y_pred = cross_val_predict(best_rf, X_scaled, y, cv=10)
rf_final_accuracy = accuracy_score(y, rf_y_pred)

print(f"\nFinal Random Forest model accuracy: {rf_final_accuracy:.4f}")
print("\nClassification report:")
print(classification_report(y, rf_y_pred))

# Confusion matrix
print("\nConfusion matrix:")
rf_conf_matrix = confusion_matrix(y, rf_y_pred)
print(rf_conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(rf_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Churn (0)', 'Churn (1)'],
            yticklabels=['Non-Churn (0)', 'Churn (1)'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.tight_layout()

# Feature importance analysis
plt.figure(figsize=(12, 8))
feature_names = X.columns
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()

# Update the model comparison section to include Random Forest
print("\n--- Model Comparison: KNN vs Logistic Regression vs Random Forest ---")
print(f"KNN Accuracy: {final_accuracy:.4f}")
print(f"Logistic Regression Accuracy: {log_final_accuracy:.4f}")
print(f"Random Forest Accuracy: {rf_final_accuracy:.4f}")

# Create a bar chart to compare model performances
plt.figure(figsize=(10, 6))
models = ['KNN', 'Logistic Regression', 'Random Forest']
accuracies = [final_accuracy, log_final_accuracy, rf_final_accuracy]
colors = ['#66b3ff', '#ff9999', '#99ff99']

plt.bar(models, accuracies, color=colors)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0.5, 1.0)  # Set y-axis to start from 0.5 for better visualization of differences
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add accuracy values on top of bars
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()

# Analyze which instances are misclassified by all models vs. only some
knn_misclassified = y_pred != y
log_misclassified = log_y_pred != y
rf_misclassified = rf_y_pred != y

all_misclassified = np.logical_and(np.logical_and(knn_misclassified, log_misclassified), rf_misclassified)
only_knn_misclassified = np.logical_and(knn_misclassified, np.logical_and(~log_misclassified, ~rf_misclassified))
only_log_misclassified = np.logical_and(~knn_misclassified, np.logical_and(log_misclassified, ~rf_misclassified))
only_rf_misclassified = np.logical_and(~knn_misclassified, np.logical_and(~log_misclassified, rf_misclassified))

print(f"\nInstances misclassified by all models: {np.sum(all_misclassified)}")
print(f"Instances misclassified only by KNN: {np.sum(only_knn_misclassified)}")
print(f"Instances misclassified only by Logistic Regression: {np.sum(only_log_misclassified)}")
print(f"Instances misclassified only by Random Forest: {np.sum(only_rf_misclassified)}")

plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')