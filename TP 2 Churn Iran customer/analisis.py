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

# Since we have many features, let's analyze feature importance indirectly
# by looking at model performance when using only pairs of features
print("\nAnalyzing feature importance for KNN model...")

# Get all pairs of features
feature_names = X.columns
feature_pairs = [(i, j) for i in range(len(feature_names)) for j in range(i+1, len(feature_names))]

# Evaluate each pair
pair_scores = []
for i, j in feature_pairs:
    # Select only these two features
    X_pair = X_scaled[:, [i, j]]
    
    # Evaluate with cross-validation
    knn_pair = KNeighborsClassifier(n_neighbors=best_k)
    score = cross_val_score(knn_pair, X_pair, y, cv=5, scoring='accuracy').mean()
    
    pair_scores.append((feature_names[i], feature_names[j], score))

# Sort pairs by score
pair_scores.sort(key=lambda x: x[2], reverse=True)

# Print top 5 feature pairs
print("\nTop 5 feature pairs for KNN prediction:")
for feat1, feat2, score in pair_scores[:5]:
    print(f"{feat1} + {feat2}: {score:.4f}")

# After finding the best feature pair and their indices
best_pair = pair_scores[0]
best_feat1, best_feat2 = best_pair[0], best_pair[1]

# Get indices of these features
idx1 = list(feature_names).index(best_feat1)
idx2 = list(feature_names).index(best_feat2)

# Create a mesh grid for the decision boundary
x_min, x_max = X_scaled[:, idx1].min() - 1, X_scaled[:, idx1].max() + 1
y_min, y_max = X_scaled[:, idx2].min() - 1, X_scaled[:, idx2].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Create a copy of the scaled data for predictions
mesh_points = np.zeros((xx.size, X_scaled.shape[1]))


# Fill the mesh with the mean values of all features
for i in range(X_scaled.shape[1]):
    mesh_points[:, i] = np.mean(X_scaled[:, i])

# Update only the two features we're plotting
mesh_points[:, idx1] = xx.ravel()
mesh_points[:, idx2] = yy.ravel()

# Use the full model to predict on the mesh grid
Z = best_knn.predict(mesh_points)
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

# Plot the training points
scatter = plt.scatter(X_scaled[:, idx1], X_scaled[:, idx2], c=y, 
                     edgecolor='k', s=50, cmap='coolwarm')

plt.xlabel(f'{best_feat1} (scaled)')
plt.ylabel(f'{best_feat2} (scaled)')
plt.title(f'KNN Decision Boundary Using Full Model\nProjected onto {best_feat1} vs {best_feat2} (k={best_k})')

# Add a legend
handles, labels = scatter.legend_elements()
plt.legend(handles, ['Non-Churn (0)', 'Churn (1)'], loc="upper right")

# Add explanation text
plt.figtext(0.5, 0.01, 
           "Note: This visualization shows the full model's decision boundary projected onto the two most important features.\nAll other features are held constant at their mean values.",
           ha="center", fontsize=9, bbox={"facecolor":"white", "alpha":0.8, "pad":5})

plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make room for the note

plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')