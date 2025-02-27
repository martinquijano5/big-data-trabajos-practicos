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

#ver la cantidad de cada especie
sums = {}
for species, count in iris_df['Species'].value_counts().items():
    sums[species] = count
print("sums: " + str(sums))

# Create a correlation matrix for the numeric features
plt.figure(figsize=(10, 8))
# Select only numeric columns for correlation, excluding Id
numeric_df = iris_df.select_dtypes(include=[np.number]).drop('Id', axis=1)
# Calculate the correlation matrix
corr_matrix = numeric_df.corr()
# Create a heatmap of the correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Iris Features')
plt.tight_layout()
    
# Create a boxplot for sepalLength, sepalWidth, petalWidth, and petalLength, separated by species
plt.figure(figsize=(12, 8))

# Boxplot for sepalLength
plt.subplot(2, 2, 1)
sns.boxplot(x='Species', y='SepalLengthCm', data=iris_df)
plt.title('Boxplot of Sepal Length by Species')

# Boxplot for sepalWidth
plt.subplot(2, 2, 2)
sns.boxplot(x='Species', y='SepalWidthCm', data=iris_df)
plt.title('Boxplot of Sepal Width by Species')

# Boxplot for petalWidth
plt.subplot(2, 2, 3)
sns.boxplot(x='Species', y='PetalWidthCm', data=iris_df)
plt.title('Boxplot of Petal Width by Species')

# Boxplot for petalLength
plt.subplot(2, 2, 4)
sns.boxplot(x='Species', y='PetalLengthCm', data=iris_df)
plt.title('Boxplot of Petal Length by Species')

plt.tight_layout()
# Don't call plt.show() here - we'll show all figures at the end

# Add KNN classification analysis with cross-validation
print("\n--- KNN Classification Analysis with Cross-Validation ---")

# Prepare the data
X = iris_df.drop(['Species', 'Id'], axis=1)  # Features (all columns except Species and Id)
y = iris_df['Species']  # Target variable

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

# Create a new figure for the cross-validation accuracy graph
plt.figure(figsize=(10, 6))
plt.plot(k_range, cv_scores)
plt.xlabel('Value of K')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Cross-Validation Accuracy for Different Values of K')
plt.grid(True)

# Find the best k value (smallest k with highest accuracy)
best_k = 0
best_accuracy = 0
for i, score in enumerate(cv_scores):
    if score > best_accuracy:
        best_accuracy = score
        best_k = k_range[i]
    elif score == best_accuracy and k_range[i] < best_k:
        best_k = k_range[i]

print(f"\nBest k value from cross-validation: {best_k} with accuracy: {best_accuracy:.4f}")

# Train the model with the best k value on the full dataset
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_scaled, y)

# Create a confusion matrix using cross-validation predictions
y_pred = cross_val_predict(best_knn, X_scaled, y, cv=10)

print("\nCross-Validation Classification Report:")
print(classification_report(y, y_pred))

print("\nCross-Validation Confusion Matrix:")
conf_matrix = confusion_matrix(y, y_pred)
print(conf_matrix)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y),
            yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Cross-Validation Confusion Matrix (k={best_k})')

# Create a pairplot to visualize the feature relationships
sns.pairplot(iris_df.drop('Id', axis=1), hue='Species', height=2.5)
plt.suptitle(f'Pairplot of Iris Features (k={best_k})', y=1.02)

# Create a decision boundary visualization with cross-validated model
plt.figure(figsize=(8, 6))  # Changed from (12, 5) to (8, 6) for a single plot

# Select only the most informative feature pair
feature1 = 'PetalLengthCm'
feature2 = 'PetalWidthCm'

# Create a mapping of species to numeric values for coloring
species_to_num = {species: i for i, species in enumerate(np.unique(y))}
y_numeric = np.array([species_to_num[species] for species in y])

# Create a mesh grid
x_min, x_max = X[feature1].min() - 0.5, X[feature1].max() + 0.5
y_min, y_max = X[feature2].min() - 0.5, X[feature2].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Create a DataFrame with all features set to their means
mesh_df = pd.DataFrame(index=range(xx.size), columns=X.columns)
for col in X.columns:
    mesh_df[col] = X[col].mean()

# Update the two features we're plotting
mesh_df[feature1] = xx.ravel()
mesh_df[feature2] = yy.ravel()

# Get predictions
Z = best_knn.predict(scaler.transform(mesh_df))
Z_numeric = np.array([species_to_num[species] for species in Z])
Z_numeric = Z_numeric.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z_numeric, alpha=0.3, cmap='viridis')

# Plot the data points, colored by their actual class
scatter = plt.scatter(X[feature1], X[feature2], c=y_numeric, 
           edgecolor='k', s=50, cmap='viridis')

# Highlight misclassified points with a red circle
misclassified = y_pred != y
plt.scatter(X[feature1][misclassified], X[feature2][misclassified], 
            s=80, facecolors='none', edgecolors='r', linewidths=2)

plt.xlabel(feature1)
plt.ylabel(feature2)
plt.title(f'Decision Boundaries for {feature1} vs {feature2} (k={best_k})')

# Add a legend
handles, labels = scatter.legend_elements()
legend = plt.legend(handles, np.unique(y), loc="upper right", title="Species")

plt.tight_layout()

# en la figure 6 se ven 5 puntos rojos. Esos son los puntos que fueron mal clasificados.

#    - Iris-setosa (violeta): Claramente separada de las otras especies
#    - Iris-versicolor (celeste): Ubicada en la región central
#    - Iris-virginica (amarillo): Ubicada en la región superior derecha

# el overlapping se da en flores que son iris-virginica pero se clasifican como iris-versicolor

# logistic regression analysis
print("\n--- Logistic Regression Analysis ---")

# Prepare the data (same as before)
X_log = iris_df.drop(['Species', 'Id'], axis=1)  # Features
y_log = iris_df['Species']  # Target variable

# Scale the features
X_log_scaled = scaler.fit_transform(X_log)

# Create and train the logistic regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# Perform cross-validation
log_cv_scores = cross_val_score(log_reg, X_log_scaled, y_log, cv=10, scoring='accuracy')
log_cv_accuracy = log_cv_scores.mean()

print(f"\nLogistic Regression Cross-Validation Accuracy: {log_cv_accuracy:.4f}")

# Get cross-validation predictions for classification report and confusion matrix
log_y_pred = cross_val_predict(log_reg, X_log_scaled, y_log, cv=10)

# Define log_misclassified here, right after getting log_y_pred
log_misclassified = log_y_pred != y_log

print("\nLogistic Regression Classification Report:")
print(classification_report(y_log, log_y_pred))

print("\nLogistic Regression Confusion Matrix:")
log_conf_matrix = confusion_matrix(y_log, log_y_pred)
print(log_conf_matrix)

# Visualize the confusion matrix for logistic regression
plt.figure(figsize=(8, 6))
sns.heatmap(log_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_log),
            yticklabels=np.unique(y_log))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression Confusion Matrix')

#la regresion logicstia tiene un poco menos de accuracy que el knn, 
# aunque cambia la matriz de confusion repartiendo los errores entre versicolor y virginica, 
# en vez de tener todos los errores en virginica como en el knn.

# Add K-means clustering analysis
print("\n--- K-means Clustering Analysis ---")

# Prepare the data (same as before, but without labels since K-means is unsupervised)
X_kmeans = iris_df.drop(['Species', 'Id'], axis=1)  # Features
y_true = iris_df['Species']  # True labels (only for evaluation)

# Scale the features
X_kmeans_scaled = scaler.fit_transform(X_kmeans)

# Create and fit the K-means model with k=3 (since we know there are 3 species)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_kmeans_scaled)

# Get the cluster assignments
cluster_labels = kmeans.labels_

# Map cluster labels to species names for better interpretation
# This is a bit tricky since K-means doesn't know the true labels
# We'll map each cluster to the most common species in that cluster
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

# Evaluate clustering performance
rand_score = adjusted_rand_score(y_true, mapped_labels)
print(f"\nAdjusted Rand Score: {rand_score:.4f}")

# Create a confusion matrix to compare K-means clusters with true species
kmeans_conf_matrix = confusion_matrix(y_true, mapped_labels)
print("\nK-means Clustering Confusion Matrix:")
print(kmeans_conf_matrix)

# Visualize the confusion matrix for K-means
plt.figure(figsize=(8, 6))
sns.heatmap(kmeans_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_true),
            yticklabels=np.unique(y_true))
plt.xlabel('K-means Cluster (mapped to most common species)')
plt.ylabel('True Species')
plt.title('K-means Clustering Confusion Matrix')

# Visualize the clusters in 2D using the most informative features
plt.figure(figsize=(10, 8))

# Select the most informative features
feature1 = 'PetalLengthCm'
feature2 = 'PetalWidthCm'

# Create a scatter plot colored by the K-means clusters
plt.subplot(1, 2, 1)
scatter_kmeans = plt.scatter(X_kmeans[feature1], X_kmeans[feature2], c=cluster_labels, 
                            cmap='viridis', edgecolor='k', s=50)
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.title('K-means Clusters')

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
scatter_true = plt.scatter(X_kmeans[feature1], X_kmeans[feature2], c=y_numeric, 
                          cmap='viridis', edgecolor='k', s=50)
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.title('True Species')

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

plt.suptitle('K-means Clusters vs. True Species', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle

#el k-means tuvo un rand score de 0.62
#la matriz de confusion para sectosa es buena, pero para las otras dos no es tan buena, en especial para virginica
#el grafico de k-means clusters vs true species nos muestra:
#k-means clusters nos muestra lo que asume el modelo
#true species nos muestra la realidad del dataset, poniendole color a los puntos en base a su especie

plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
# This will be executed after the user presses Enter
print("Programa finalizado. Cerrando figuras...")
plt.close('all')  # Close all figures
