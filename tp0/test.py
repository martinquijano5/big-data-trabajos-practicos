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
k_range = range(1, 31)
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
sns.pairplot(iris_df, hue='Species', height=2.5)
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

#la regresion logicstia tiene un poco menos de accuracy que el knn, aunque cambia la matriz de confusion repartiendo los errores entre versicolor y virginica, en vez de tener todos los errores en virginica como en el knn.

plt.show(block=False)

input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")

# This will be executed after the user presses Enter
print("Programa finalizado. Cerrando figuras...")
plt.close('all')  # Close all figures
