import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
from collections import Counter

# Function to find the best k for KNN
def find_best_k(X, y, k_range=range(1, 51)):
    cv_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())
    
    # Find the best k value (smallest k with highest accuracy)
    best_k_idx = np.argmax(cv_scores)
    best_k = k_range[best_k_idx]
    best_score = cv_scores[best_k_idx]
    
    return best_k, best_score

# Function to evaluate KNN model using cross-validation
def evaluate_knn_cv(X, y, k, cv=10):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')
    return scores.mean()

# Function to evaluate Logistic Regression model using cross-validation
def evaluate_logreg_cv(X, y, cv=10):
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(log_reg, X, y, cv=cv, scoring='accuracy')
    return scores.mean()

# Function to evaluate KMeans model using cross-validation
def evaluate_kmeans_cv(X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    accuracies = []
    ari_scores = []
    
    for train_index, test_index in kf.split(X):
        # We use the entire dataset for clustering, but evaluate on the test fold
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X)
        cluster_labels = kmeans.labels_
        
        # Map cluster labels to actual species labels
        y_numeric = pd.factorize(y)[0]
        
        # Create a mapping from cluster labels to true labels
        cluster_to_species = {}
        for cluster_label in range(3):
            mask = (cluster_labels == cluster_label)
            if np.sum(mask) > 0:  # Ensure the cluster has points
                true_labels = y_numeric[mask]
                counter = Counter(true_labels)
                most_common_label = counter.most_common(1)[0][0]
                cluster_to_species[cluster_label] = most_common_label
        
        # Map the cluster labels to species labels
        mapped_labels = np.array([cluster_to_species.get(label, -1) for label in cluster_labels])
        
        # Calculate accuracy and ARI for this fold
        test_accuracy = np.mean(mapped_labels[test_index] == y_numeric[test_index])
        test_ari = adjusted_rand_score(y_numeric[test_index], cluster_labels[test_index])
        
        accuracies.append(test_accuracy)
        ari_scores.append(test_ari)
    
    return np.mean(accuracies), np.mean(ari_scores)

# Load the dataset
iris_df = pd.read_csv('tp0/iris.csv')
print("Dataset loaded successfully.")

# Extract features and target
X_original = iris_df.drop(['Species', 'Id'], axis=1)
y = iris_df['Species']

# Create a dictionary to store results
results = {
    'Transformation': [],
    'KNN_Accuracy': [],
    'KNN_Best_K': [],
    'LogReg_Accuracy': [],
    'KMeans_Accuracy': [],
    'KMeans_ARI': []
}

# 1. Original data with StandardScaler (baseline)
print("\nEvaluating Original Data with StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_original)

# KNN with cross-validation
best_k, best_score = find_best_k(X_scaled, y)
knn_accuracy = evaluate_knn_cv(X_scaled, y, best_k)

# Logistic Regression with cross-validation
logreg_accuracy = evaluate_logreg_cv(X_scaled, y)

# KMeans with cross-validation
kmeans_accuracy, kmeans_ari = evaluate_kmeans_cv(X_scaled, y)

# Store results
results['Transformation'].append('Original')
results['KNN_Accuracy'].append(knn_accuracy)
results['KNN_Best_K'].append(best_k)
results['LogReg_Accuracy'].append(logreg_accuracy)
results['KMeans_Accuracy'].append(kmeans_accuracy)
results['KMeans_ARI'].append(kmeans_ari)

# 2. MinMax Scaling
print("\nEvaluating MinMax Scaling...")
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X_original)

# KNN with cross-validation
best_k, best_score = find_best_k(X_minmax, y)
knn_accuracy = evaluate_knn_cv(X_minmax, y, best_k)

# Logistic Regression with cross-validation
logreg_accuracy = evaluate_logreg_cv(X_minmax, y)

# KMeans with cross-validation
kmeans_accuracy, kmeans_ari = evaluate_kmeans_cv(X_minmax, y)

# Store results
results['Transformation'].append('MinMax Scaling')
results['KNN_Accuracy'].append(knn_accuracy)
results['KNN_Best_K'].append(best_k)
results['LogReg_Accuracy'].append(logreg_accuracy)
results['KMeans_Accuracy'].append(kmeans_accuracy)
results['KMeans_ARI'].append(kmeans_ari)

# 3. Logarithmic Transformation
print("\nEvaluating Logarithmic Transformation...")
# Apply log transformation (adding a small constant to avoid log(0))
X_log = np.log1p(X_original)

# Apply standard scaling after log transformation
scaler_log = StandardScaler()
X_log_scaled = scaler_log.fit_transform(X_log)

# KNN with cross-validation
best_k, best_score = find_best_k(X_log_scaled, y)
knn_accuracy = evaluate_knn_cv(X_log_scaled, y, best_k)

# Logistic Regression with cross-validation
logreg_accuracy = evaluate_logreg_cv(X_log_scaled, y)

# KMeans with cross-validation
kmeans_accuracy, kmeans_ari = evaluate_kmeans_cv(X_log_scaled, y)

# Store results
results['Transformation'].append('Logarithmic')
results['KNN_Accuracy'].append(knn_accuracy)
results['KNN_Best_K'].append(best_k)
results['LogReg_Accuracy'].append(logreg_accuracy)
results['KMeans_Accuracy'].append(kmeans_accuracy)
results['KMeans_ARI'].append(kmeans_ari)

# 4. Average Sepal and Petal
print("\nEvaluating Average Sepal and Petal...")
# Create new features
iris_with_avg = iris_df.copy()
iris_with_avg['AvgSepalSize'] = (iris_df['SepalLengthCm'] + iris_df['SepalWidthCm']) / 2
iris_with_avg['AvgPetalSize'] = (iris_df['PetalLengthCm'] + iris_df['PetalWidthCm']) / 2

# Extract features
X_avg_sepal_petal = iris_with_avg[['AvgSepalSize', 'AvgPetalSize']]

# Apply standard scaling
scaler_avg_sp = StandardScaler()
X_avg_sp_scaled = scaler_avg_sp.fit_transform(X_avg_sepal_petal)

# KNN with cross-validation
best_k, best_score = find_best_k(X_avg_sp_scaled, y)
knn_accuracy = evaluate_knn_cv(X_avg_sp_scaled, y, best_k)

# Logistic Regression with cross-validation
logreg_accuracy = evaluate_logreg_cv(X_avg_sp_scaled, y)

# KMeans with cross-validation
kmeans_accuracy, kmeans_ari = evaluate_kmeans_cv(X_avg_sp_scaled, y)

# Store results
results['Transformation'].append('Avg Sepal & Petal')
results['KNN_Accuracy'].append(knn_accuracy)
results['KNN_Best_K'].append(best_k)
results['LogReg_Accuracy'].append(logreg_accuracy)
results['KMeans_Accuracy'].append(kmeans_accuracy)
results['KMeans_ARI'].append(kmeans_ari)

# 5. Average Height and Width
print("\nEvaluating Average Height and Width...")
# Create new features
iris_with_hw = iris_df.copy()
iris_with_hw['AvgHeight'] = (iris_df['SepalLengthCm'] + iris_df['PetalLengthCm']) / 2
iris_with_hw['AvgWidth'] = (iris_df['SepalWidthCm'] + iris_df['PetalWidthCm']) / 2

# Extract features
X_avg_hw = iris_with_hw[['AvgHeight', 'AvgWidth']]

# Apply standard scaling
scaler_avg_hw = StandardScaler()
X_avg_hw_scaled = scaler_avg_hw.fit_transform(X_avg_hw)

# KNN with cross-validation
best_k, best_score = find_best_k(X_avg_hw_scaled, y)
knn_accuracy = evaluate_knn_cv(X_avg_hw_scaled, y, best_k)

# Logistic Regression with cross-validation
logreg_accuracy = evaluate_logreg_cv(X_avg_hw_scaled, y)

# KMeans with cross-validation
kmeans_accuracy, kmeans_ari = evaluate_kmeans_cv(X_avg_hw_scaled, y)

# Store results
results['Transformation'].append('Avg Height & Width')
results['KNN_Accuracy'].append(knn_accuracy)
results['KNN_Best_K'].append(best_k)
results['LogReg_Accuracy'].append(logreg_accuracy)
results['KMeans_Accuracy'].append(kmeans_accuracy)
results['KMeans_ARI'].append(kmeans_ari)

# 6. Ratios
print("\nEvaluating Ratios...")
# Create new features
iris_with_ratios = iris_df.copy()
iris_with_ratios['SepalRatio'] = iris_df['SepalLengthCm'] / iris_df['SepalWidthCm']
iris_with_ratios['PetalRatio'] = iris_df['PetalLengthCm'] / iris_df['PetalWidthCm']
iris_with_ratios['LengthRatio'] = iris_df['SepalLengthCm'] / iris_df['PetalLengthCm']
iris_with_ratios['WidthRatio'] = iris_df['SepalWidthCm'] / iris_df['PetalWidthCm']

# Extract features
X_ratios = iris_with_ratios[['SepalRatio', 'PetalRatio', 'LengthRatio', 'WidthRatio']]

# Apply standard scaling
scaler_ratios = StandardScaler()
X_ratios_scaled = scaler_ratios.fit_transform(X_ratios)

# KNN with cross-validation
best_k, best_score = find_best_k(X_ratios_scaled, y)
knn_accuracy = evaluate_knn_cv(X_ratios_scaled, y, best_k)

# Logistic Regression with cross-validation
logreg_accuracy = evaluate_logreg_cv(X_ratios_scaled, y)

# KMeans with cross-validation
kmeans_accuracy, kmeans_ari = evaluate_kmeans_cv(X_ratios_scaled, y)

# Store results
results['Transformation'].append('Ratios')
results['KNN_Accuracy'].append(knn_accuracy)
results['KNN_Best_K'].append(best_k)
results['LogReg_Accuracy'].append(logreg_accuracy)
results['KMeans_Accuracy'].append(kmeans_accuracy)
results['KMeans_ARI'].append(kmeans_ari)

# Create a DataFrame with the results
results_df = pd.DataFrame(results)
print("\nResults Summary:")
print(results_df.to_string())

# Find the best transformation for each model
best_knn = results_df.loc[results_df['KNN_Accuracy'].idxmax()]
best_logreg = results_df.loc[results_df['LogReg_Accuracy'].idxmax()]
best_kmeans = results_df.loc[results_df['KMeans_Accuracy'].idxmax()]
best_kmeans_ari = results_df.loc[results_df['KMeans_ARI'].idxmax()]

print("\nBest Transformation for KNN:")
print(f"Transformation: {best_knn['Transformation']}")
print(f"Accuracy: {best_knn['KNN_Accuracy']:.4f}")
print(f"Best K: {best_knn['KNN_Best_K']}")

print("\nBest Transformation for Logistic Regression:")
print(f"Transformation: {best_logreg['Transformation']}")
print(f"Accuracy: {best_logreg['LogReg_Accuracy']:.4f}")

print("\nBest Transformation for KMeans (by Accuracy):")
print(f"Transformation: {best_kmeans['Transformation']}")
print(f"Accuracy: {best_kmeans['KMeans_Accuracy']:.4f}")
print(f"ARI: {best_kmeans['KMeans_ARI']:.4f}")

print("\nBest Transformation for KMeans (by Adjusted Rand Index):")
print(f"Transformation: {best_kmeans_ari['Transformation']}")
print(f"Accuracy: {best_kmeans_ari['KMeans_Accuracy']:.4f}")
print(f"ARI: {best_kmeans_ari['KMeans_ARI']:.4f}")

# Create bar plots to visualize the results
plt.figure(figsize=(20, 14))  # Increased figure size for better readability

# KNN Accuracy
plt.subplot(2, 2, 1)
# Find max value and create color list
max_knn = results_df['KNN_Accuracy'].max()
knn_colors = ['#ff7f0e' if results_df['KNN_Accuracy'][i] == max_knn else '#1f77b4' 
              for i in range(len(results_df))]
ax1 = sns.barplot(x='Transformation', y='KNN_Accuracy', data=results_df, palette=knn_colors)
plt.title('KNN Accuracy by Transformation', fontsize=14)
plt.xticks(rotation=30, ha='right', fontsize=10)  # Angled labels for better readability
plt.ylim(0, 1.05)  # Increased upper limit to make room for text
# Add text labels with consistent formatting
for i, p in enumerate(ax1.patches):
    ax1.annotate(f'{p.get_height():.2f}', 
                (p.get_x() + p.get_width() / 2., p.get_height() + 0.02), 
                ha='center', va='bottom', fontsize=10)

# Logistic Regression Accuracy
plt.subplot(2, 2, 2)
# Find max value and create color list
max_logreg = results_df['LogReg_Accuracy'].max()
logreg_colors = ['#ff7f0e' if results_df['LogReg_Accuracy'][i] == max_logreg else '#1f77b4' 
                for i in range(len(results_df))]
ax2 = sns.barplot(x='Transformation', y='LogReg_Accuracy', data=results_df, palette=logreg_colors)
plt.title('Logistic Regression Accuracy by Transformation', fontsize=14)
plt.xticks(rotation=30, ha='right', fontsize=10)
plt.ylim(0, 1.05)  # Increased upper limit to make room for text
# Add text labels
for i, p in enumerate(ax2.patches):
    ax2.annotate(f'{p.get_height():.2f}', 
                (p.get_x() + p.get_width() / 2., p.get_height() + 0.02), 
                ha='center', va='bottom', fontsize=10)

# KMeans Accuracy
plt.subplot(2, 2, 3)
# Find max value and create color list
max_kmeans = results_df['KMeans_Accuracy'].max()
kmeans_colors = ['#ff7f0e' if results_df['KMeans_Accuracy'][i] == max_kmeans else '#1f77b4' 
                for i in range(len(results_df))]
ax3 = sns.barplot(x='Transformation', y='KMeans_Accuracy', data=results_df, palette=kmeans_colors)
plt.title('KMeans Accuracy by Transformation', fontsize=14)
plt.xticks(rotation=30, ha='right', fontsize=10)
plt.ylim(0, 1.05)  # Increased upper limit to make room for text
# Add text labels
for i, p in enumerate(ax3.patches):
    ax3.annotate(f'{p.get_height():.2f}', 
                (p.get_x() + p.get_width() / 2., p.get_height() + 0.02), 
                ha='center', va='bottom', fontsize=10)

# KMeans ARI
plt.subplot(2, 2, 4)
# Find max value and create color list
max_ari = results_df['KMeans_ARI'].max()
ari_colors = ['#ff7f0e' if results_df['KMeans_ARI'][i] == max_ari else '#1f77b4' 
             for i in range(len(results_df))]
ax4 = sns.barplot(x='Transformation', y='KMeans_ARI', data=results_df, palette=ari_colors)
plt.title('KMeans Adjusted Rand Index by Transformation', fontsize=14)
plt.xticks(rotation=30, ha='right', fontsize=10)
plt.ylim(0, 1.05)  # Increased upper limit to make room for text
# Add text labels
for i, p in enumerate(ax4.patches):
    ax4.annotate(f'{p.get_height():.2f}', 
                (p.get_x() + p.get_width() / 2., p.get_height() + 0.02), 
                ha='center', va='bottom', fontsize=10)

plt.tight_layout(pad=3.0)  # Add more padding between subplots

plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
# This will be executed after the user presses Enter
print("Programa finalizado. Cerrando figuras...")
plt.close('all')  # Close all figures
