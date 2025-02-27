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
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.stats import mode

iris_df = pd.read_csv('tp0/iris.csv')

print("head:")
print(iris_df.head())

print("info:")
print(iris_df.info())

print("describe:")
print(iris_df.describe())

# Create new features using ratios
iris_df['SepalRatio'] = iris_df['SepalLengthCm'] / iris_df['SepalWidthCm']
iris_df['PetalRatio'] = iris_df['PetalLengthCm'] / iris_df['PetalWidthCm']
iris_df['LengthRatio'] = iris_df['SepalLengthCm'] / iris_df['PetalLengthCm']
iris_df['WidthRatio'] = iris_df['SepalWidthCm'] / iris_df['PetalWidthCm']

# Print the head of the dataframe with new features
print("\nDataframe with ratio features:")
print(iris_df.head())

#ver la cantidad de cada especie
sums = {}
for species, count in iris_df['Species'].value_counts().items():
    sums[species] = count
print("sums: " + str(sums))
    
# Create a correlation matrix for only the ratio features
plt.figure(figsize=(10, 8))
# Select only the ratio features for correlation
ratio_features_df = iris_df[['SepalRatio', 'PetalRatio', 'LengthRatio', 'WidthRatio']]
# Calculate the correlation matrix
corr_matrix = ratio_features_df.corr()
# Create a heatmap of the correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Ratio Features')
plt.tight_layout()

# Create a boxplot for only the new ratio features
plt.figure(figsize=(15, 10))

# Boxplots for new ratio features
plt.subplot(2, 2, 1)
sns.boxplot(x='Species', y='SepalRatio', data=iris_df)
plt.title('Boxplot of Sepal Ratio by Species')

plt.subplot(2, 2, 2)
sns.boxplot(x='Species', y='PetalRatio', data=iris_df)
plt.title('Boxplot of Petal Ratio by Species')

plt.subplot(2, 2, 3)
sns.boxplot(x='Species', y='LengthRatio', data=iris_df)
plt.title('Boxplot of Length Ratio by Species')

plt.subplot(2, 2, 4)
sns.boxplot(x='Species', y='WidthRatio', data=iris_df)
plt.title('Boxplot of Width Ratio by Species')

plt.tight_layout()

# Add KNN classification analysis with cross-validation
print("\n--- KNN Classification Analysis with Cross-Validation ---")

# Prepare the data with ratio features
X = iris_df[['SepalRatio', 'PetalRatio', 'LengthRatio', 'WidthRatio']]  # Using only the ratio features
y = iris_df['Species']  # Target variable

# For comparison, also prepare the original features
X_original = iris_df.drop(['Species', 'Id', 'SepalRatio', 'PetalRatio', 'LengthRatio', 'WidthRatio'], axis=1)

# Scale both feature sets with separate scalers
scaler_ratio = StandardScaler().fit(X)
scaler_orig = StandardScaler().fit(X_original)
X_scaled = scaler_ratio.transform(X)
X_original_scaled = scaler_orig.transform(X_original)

# Find the optimal k value using cross-validation for ratio features
k_range = range(1, 51)
cv_scores = []
cv_scores_original = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # Cross-validation with ratio features
    scores = cross_val_score(knn, X_scaled, y, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
    
    # Cross-validation with original features for comparison
    scores_original = cross_val_score(knn, X_original_scaled, y, cv=10, scoring='accuracy')
    cv_scores_original.append(scores_original.mean())

# Create a figure for the cross-validation accuracy comparison
plt.figure(figsize=(10, 6))
plt.plot(k_range, cv_scores, label='Ratio Features')
plt.plot(k_range, cv_scores_original, label='Original Features')
plt.xlabel('Value of K')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Cross-Validation Accuracy Comparison')
plt.legend()
plt.grid(True)

# Find the best k value for ratio features
best_k = 0
best_accuracy = 0
for i, score in enumerate(cv_scores):
    if score > best_accuracy:
        best_accuracy = score
        best_k = k_range[i]
    elif score == best_accuracy and k_range[i] < best_k:
        best_k = k_range[i]

# Find the best k value for original features
best_k_original = 0
best_accuracy_original = 0
for i, score in enumerate(cv_scores_original):
    if score > best_accuracy_original:
        best_accuracy_original = score
        best_k_original = k_range[i]
    elif score == best_accuracy_original and k_range[i] < best_k_original:
        best_k_original = k_range[i]

print(f"\nBest k value for ratio features: {best_k} with accuracy: {best_accuracy:.4f}")
print(f"Best k value for original features: {best_k_original} with accuracy: {best_accuracy_original:.4f}")

# Train the model with the best k value on the ratio features
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_scaled, y)

# Create a confusion matrix using cross-validation predictions for ratio features
y_pred = cross_val_predict(best_knn, X_scaled, y, cv=10)

print("\nCross-Validation Classification Report (Ratio Features):")
print(classification_report(y, y_pred))

print("\nCross-Validation Confusion Matrix (Ratio Features):")
conf_matrix = confusion_matrix(y, y_pred)
print(conf_matrix)

# Also get results for original features for comparison
best_knn_original = KNeighborsClassifier(n_neighbors=best_k_original)
y_pred_original = cross_val_predict(best_knn_original, X_original_scaled, y, cv=10)

print("\nCross-Validation Classification Report (Original Features):")
print(classification_report(y, y_pred_original))

print("\nCross-Validation Confusion Matrix (Original Features):")
conf_matrix_original = confusion_matrix(y, y_pred_original)
print(conf_matrix_original)

# Visualize the confusion matrices side by side
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y),
            yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - Ratio Features (k={best_k})')

plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_original, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y),
            yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - Original Features (k={best_k_original})')

plt.tight_layout()

# Create a decision boundary visualization with the ratio features
# We'll use the two most important ratio features for visualization
plt.figure(figsize=(10, 8))

# Select the two most important ratio features based on boxplot separation
feature1 = 'LengthRatio'
feature2 = 'WidthRatio'
X_viz = iris_df[[feature1, feature2]]

# Create a mesh grid for the selected ratio features
x_min, x_max = X_viz[feature1].min() - 0.5, X_viz[feature1].max() + 0.5
y_min, y_max = X_viz[feature2].min() - 0.5, X_viz[feature2].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Train a KNN model with just these two features
scaler_viz = StandardScaler().fit(X_viz)  # Create a separate scaler for visualization
X_viz_scaled = scaler_viz.transform(X_viz)
knn_viz = KNeighborsClassifier(n_neighbors=best_k)
knn_viz.fit(X_viz_scaled, y)

# Get predictions
mesh_points = np.c_[xx.ravel(), yy.ravel()]
Z = knn_viz.predict(scaler_viz.transform(mesh_points))  # Use the visualization scaler

# Map species to numeric values for coloring
species_to_num = {species: i for i, species in enumerate(np.unique(y))}
Z_numeric = np.array([species_to_num[species] for species in Z])
Z_numeric = Z_numeric.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z_numeric, alpha=0.3, cmap='viridis')

# Plot the data points, colored by their actual class
y_numeric = np.array([species_to_num[species] for species in y])
scatter = plt.scatter(X_viz[feature1], X_viz[feature2], c=y_numeric, 
           edgecolor='k', s=50, cmap='viridis')

# Get predictions for the visualization dataset
y_pred_viz = cross_val_predict(knn_viz, X_viz_scaled, y, cv=10)

# Highlight misclassified points with a red circle
misclassified = y_pred_viz != y
plt.scatter(X_viz[feature1][misclassified], X_viz[feature2][misclassified], 
            s=80, facecolors='none', edgecolors='r', linewidths=2)

plt.xlabel(feature1)
plt.ylabel(feature2)
plt.title(f'Decision Boundaries with Selected Ratio Features (k={best_k})')

# Add a legend
handles, labels = scatter.legend_elements()
legend = plt.legend(handles, np.unique(y), loc="upper right", title="Species")

plt.tight_layout()

# logistic regression analysis
print("\n--- Logistic Regression Analysis ---")

# Prepare the data with ratio features
X_log = iris_df[['SepalRatio', 'PetalRatio', 'LengthRatio', 'WidthRatio']]  # Using only the ratio features
y_log = iris_df['Species']  # Target variable

# Scale the features - use the same X_log for fitting and transforming
X_log_scaled = scaler_ratio.transform(X_log)

# Create and train the logistic regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# Perform cross-validation with ratio features
log_cv_scores = cross_val_score(log_reg, X_log_scaled, y_log, cv=10, scoring='accuracy')
log_cv_accuracy = log_cv_scores.mean()

# Also perform cross-validation with original features for comparison
log_cv_scores_original = cross_val_score(log_reg, scaler_orig.transform(X_original), y_log, cv=10, scoring='accuracy')
log_cv_accuracy_original = log_cv_scores_original.mean()

print(f"\nLogistic Regression Cross-Validation Accuracy (Ratio Features): {log_cv_accuracy:.4f}")
print(f"Logistic Regression Cross-Validation Accuracy (Original Features): {log_cv_accuracy_original:.4f}")

# Get cross-validation predictions for classification report and confusion matrix
log_y_pred = cross_val_predict(log_reg, X_log_scaled, y_log, cv=10)
log_y_pred_original = cross_val_predict(log_reg, scaler_orig.transform(X_original), y_log, cv=10)

# Define log_misclassified for both feature sets
log_misclassified = log_y_pred != y_log
log_misclassified_original = log_y_pred_original != y_log

print("\nLogistic Regression Classification Report (Ratio Features):")
print(classification_report(y_log, log_y_pred))

print("\nLogistic Regression Classification Report (Original Features):")
print(classification_report(y_log, log_y_pred_original))

print("\nLogistic Regression Confusion Matrix (Ratio Features):")
log_conf_matrix = confusion_matrix(y_log, log_y_pred)
print(log_conf_matrix)

print("\nLogistic Regression Confusion Matrix (Original Features):")
log_conf_matrix_original = confusion_matrix(y_log, log_y_pred_original)
print(log_conf_matrix_original)

# Visualize the confusion matrices side by side
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.heatmap(log_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_log),
            yticklabels=np.unique(y_log))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression Confusion Matrix - Ratio Features')

plt.subplot(1, 2, 2)
sns.heatmap(log_conf_matrix_original, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_log),
            yticklabels=np.unique(y_log))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression Confusion Matrix - Original Features')

plt.tight_layout()

# Add K-means clustering analysis
print("\n--- K-means Clustering Analysis ---")

# Prepare the data with ratio features
X_kmeans = iris_df[['SepalRatio', 'PetalRatio', 'LengthRatio', 'WidthRatio']]  # Using only the ratio features
y_true = iris_df['Species']  # True labels (only for evaluation)

# Scale the features
X_kmeans_scaled = scaler_ratio.transform(X_kmeans)

# Create and fit the K-means model with k=3 (since we know there are 3 species)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_kmeans_scaled)

# Get the cluster assignments
cluster_labels = kmeans.labels_

# Map cluster labels to species names for better interpretation
mapped_labels = np.zeros_like(cluster_labels, dtype=object)
for cluster_id in range(3):
    cluster_mask = (cluster_labels == cluster_id)
    if np.any(cluster_mask):
        # Get the species in this cluster and count them
        species_in_cluster = y_true[cluster_mask]
        unique_species, counts = np.unique(species_in_cluster, return_counts=True)
        # Find the most common species
        most_common_species = unique_species[np.argmax(counts)]
        mapped_labels[cluster_mask] = most_common_species

# Evaluate clustering performance with ratio features
rand_score = adjusted_rand_score(y_true, mapped_labels)
print(f"\nAdjusted Rand Score (Ratio Features): {rand_score:.4f}")

# Also perform K-means with original features for comparison
X_kmeans_original = iris_df.drop(['Species', 'Id', 'SepalRatio', 'PetalRatio', 'LengthRatio', 'WidthRatio'], axis=1)
X_kmeans_original_scaled = scaler_orig.transform(X_kmeans_original)

kmeans_original = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_original.fit(X_kmeans_original_scaled)
cluster_labels_original = kmeans_original.labels_

# Map cluster labels to species names for original features
mapped_labels_original = np.zeros_like(cluster_labels_original, dtype=object)
for cluster_id in range(3):
    cluster_mask = (cluster_labels_original == cluster_id)
    if np.any(cluster_mask):
        species_in_cluster = y_true[cluster_mask]
        unique_species, counts = np.unique(species_in_cluster, return_counts=True)
        most_common_species = unique_species[np.argmax(counts)]
        mapped_labels_original[cluster_mask] = most_common_species

# Evaluate clustering performance with original features
rand_score_original = adjusted_rand_score(y_true, mapped_labels_original)
print(f"Adjusted Rand Score (Original Features): {rand_score_original:.4f}")

# Create confusion matrices for both feature sets
kmeans_conf_matrix = confusion_matrix(y_true, mapped_labels)
kmeans_conf_matrix_original = confusion_matrix(y_true, mapped_labels_original)

print("\nK-means Clustering Confusion Matrix (Ratio Features):")
print(kmeans_conf_matrix)

print("\nK-means Clustering Confusion Matrix (Original Features):")
print(kmeans_conf_matrix_original)

# Visualize the confusion matrices side by side
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.heatmap(kmeans_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_true),
            yticklabels=np.unique(y_true))
plt.xlabel('K-means Cluster (mapped to most common species)')
plt.ylabel('True Species')
plt.title('K-means Confusion Matrix - Ratio Features')

plt.subplot(1, 2, 2)
sns.heatmap(kmeans_conf_matrix_original, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_true),
            yticklabels=np.unique(y_true))
plt.xlabel('K-means Cluster (mapped to most common species)')
plt.ylabel('True Species')
plt.title('K-means Confusion Matrix - Original Features')

plt.tight_layout()

# Visualize the clusters in 2D using the two most important ratio features
plt.figure(figsize=(16, 8))

# Select the two most important ratio features for visualization
feature1 = 'LengthRatio'
feature2 = 'WidthRatio'
X_viz = iris_df[[feature1, feature2]]

# Create a scatter plot colored by the K-means clusters
plt.subplot(1, 2, 1)
scatter_kmeans = plt.scatter(X_viz[feature1], X_viz[feature2], c=cluster_labels, 
                            cmap='viridis', edgecolor='k', s=50)
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.title('K-means Clusters (Selected Ratio Features)')

# Add a legend for the K-means clusters
cluster_legend = plt.legend(*scatter_kmeans.legend_elements(),
                           title="Clusters", loc="upper right")
plt.gca().add_artist(cluster_legend)

# Add a text annotation showing which cluster maps to which species
cluster_to_species = {}
for cluster_id in range(3):
    cluster_mask = (cluster_labels == cluster_id)
    if np.any(cluster_mask):
        species_in_cluster = y_true[cluster_mask]
        unique_species, counts = np.unique(species_in_cluster, return_counts=True)
        most_common_species = unique_species[np.argmax(counts)]
        cluster_to_species[cluster_id] = most_common_species

# Add a text box with the mapping
mapping_text = "Cluster mapping:\n"
for cluster_id, species in cluster_to_species.items():
    mapping_text += f"Cluster {cluster_id} → {species}\n"
plt.annotate(mapping_text, xy=(0.05, 0.05), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

# Create a scatter plot colored by the true species for comparison
plt.subplot(1, 2, 2)
species_to_num = {species: i for i, species in enumerate(np.unique(y_true))}
y_numeric = np.array([species_to_num[species] for species in y_true])
scatter_true = plt.scatter(X_viz[feature1], X_viz[feature2], c=y_numeric, 
                          cmap='viridis', edgecolor='k', s=50)
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.title('True Species (Selected Ratio Features)')

# Add a legend for the true species
species_legend = plt.legend(*scatter_true.legend_elements(),
                           title="Species", loc="upper right")
plt.gca().add_artist(species_legend)

# Add species labels in the same format as the cluster mapping
species_text = "Species mapping:\n"
for i, species in enumerate(np.unique(y_true)):
    species_text += f"Species {i} → {species}\n"
plt.annotate(species_text, xy=(0.05, 0.05), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

plt.suptitle('K-means Clusters vs. True Species (Selected Ratio Features)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle

# Add a summary section comparing all models with both feature sets
print("\n--- Model Performance Summary ---")
print(f"KNN (Ratio Features): {best_accuracy:.4f}")
print(f"KNN (Original Features): {best_accuracy_original:.4f}")
print(f"Logistic Regression (Ratio Features): {log_cv_accuracy:.4f}")
print(f"Logistic Regression (Original Features): {log_cv_accuracy_original:.4f}")
print(f"K-means (Ratio Features): {rand_score:.4f}")
print(f"K-means (Original Features): {rand_score_original:.4f}")

# Create a bar chart comparing model performances
plt.figure(figsize=(12, 6))
models = ['KNN', 'Logistic Regression', 'K-means']
ratio_scores = [best_accuracy, log_cv_accuracy, rand_score]
orig_scores = [best_accuracy_original, log_cv_accuracy_original, rand_score_original]

x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, ratio_scores, width, label='Ratio Features')
plt.bar(x + width/2, orig_scores, width, label='Original Features')

plt.xlabel('Model')
plt.ylabel('Accuracy / Adjusted Rand Score')
plt.title('Model Performance Comparison')
plt.xticks(x, models)
plt.legend()
plt.ylim(0, 1.0)  # Set y-axis from 0 to 1

for i, v in enumerate(ratio_scores):
    plt.text(i - width/2, v + 0.01, f'{v:.4f}', ha='center')
    
for i, v in enumerate(orig_scores):
    plt.text(i + width/2, v + 0.01, f'{v:.4f}', ha='center')

plt.tight_layout()

plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
# This will be executed after the user presses Enter
print("Programa finalizado. Cerrando figuras...")
plt.close('all')  # Close all figures 