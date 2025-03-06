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

# Pairplot de las variables menos id para ver como se distribuyen las variables
sns.pairplot(iris_df.drop('Id', axis=1), hue='Species', height=2.5)
plt.suptitle(f'Pairplot of Iris Features', y=1.02)

# Matrix de correlacion con mapa de calor para ver como se relacionan las variables
plt.figure(figsize=(10, 8))
numeric_df = iris_df.select_dtypes(include=[np.number]).drop('Id', axis=1)
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Iris Features')
plt.tight_layout()
    
# Boxplot para ver como se distribuyen las variables por especie
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
sns.boxplot(x='Species', y='SepalLengthCm', data=iris_df)
plt.title('Boxplot of Sepal Length by Species')

plt.subplot(2, 2, 2)
sns.boxplot(x='Species', y='SepalWidthCm', data=iris_df)
plt.title('Boxplot of Sepal Width by Species')

plt.subplot(2, 2, 3)
sns.boxplot(x='Species', y='PetalWidthCm', data=iris_df)
plt.title('Boxplot of Petal Width by Species')

plt.subplot(2, 2, 4)
sns.boxplot(x='Species', y='PetalLengthCm', data=iris_df)
plt.title('Boxplot of Petal Length by Species')


# Analisis de clasificacion con KNN con cross-validation
print("\n--- KNN Classification Analysis con Cross-Validation ---")

X = iris_df.drop(['Species', 'Id'], axis=1)
y = iris_df['Species']  # variable objetivo

# Escalado de las variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encontrar el valor optimo de k usando cross-validation
k_range = range(1, 51)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y, cv=10, scoring='accuracy')  # 10-fold cross-validation
    cv_scores.append(scores.mean())

# Crear una nueva figura para el grafico de accuracy de cross-validation
plt.figure(figsize=(10, 6))
plt.plot(k_range, cv_scores)
plt.xlabel('Value of K')
plt.ylabel('KNN Accuracy')
plt.title('KNN Accuracy for Different Values of K')
plt.grid(True)

# Encontrar el valor optimo de k (el menor k con la mayor accuracy)
best_k = 0
best_accuracy = 0
for i, score in enumerate(cv_scores):
    if score > best_accuracy:
        best_accuracy = score
        best_k = k_range[i]
    elif score == best_accuracy and k_range[i] < best_k:
        best_k = k_range[i]

print(f"\nBest k value from cross-validation: {best_k} with accuracy: {best_accuracy:.4f}")

# Entrenar el modelo con el valor optimo de k en el dataset completo
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_scaled, y)

# Obtener predicciones del modelo final
y_pred = best_knn.predict(X_scaled)
misclassified = y_pred != y
final_accuracy = accuracy_score(y, y_pred)

print(f"\nAccuracy del modelo final: {final_accuracy:.4f}")
print("\nReporte de clasificacion del modelo final:")
print(classification_report(y, y_pred))

# matriz de confusion de knn
print("\nMatriz de confusion del modelo final:")
conf_matrix = confusion_matrix(y, y_pred)
print(conf_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y),
            yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Matriz de confusion de knn (k={best_k}')

# Grafico de fronteras de decision para el modelo final
plt.figure(figsize=(8, 6))

# Seleccionar las dos variables mas relevantes para el modelo final
feature1 = 'PetalLengthCm'
feature2 = 'PetalWidthCm'

# Crear un mapping de especies a valores numericos para colorear
species_to_num = {species: i for i, species in enumerate(np.unique(y))}
y_numeric = np.array([species_to_num[species] for species in y])

# Crear una grilla
x_min, x_max = X[feature1].min() - 0.5, X[feature1].max() + 0.5
y_min, y_max = X[feature2].min() - 0.5, X[feature2].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Crear un DataFrame con todas las features establecidas en su media
mesh_df = pd.DataFrame(index=range(xx.size), columns=X.columns)
for col in X.columns:
    mesh_df[col] = X[col].mean()

# Actualizar las dos features que estamos graficando
mesh_df[feature1] = xx.ravel()
mesh_df[feature2] = yy.ravel()

# Obtener predicciones
Z = best_knn.predict(scaler.transform(mesh_df))
Z_numeric = np.array([species_to_num[species] for species in Z])
Z_numeric = Z_numeric.reshape(xx.shape)

# Graficar la frontera de decision
plt.contourf(xx, yy, Z_numeric, alpha=0.3, cmap='viridis')

# Graficar los puntos de datos, coloreados por su clase actual
scatter = plt.scatter(X[feature1], X[feature2], c=y_numeric, 
           edgecolor='k', s=50, cmap='viridis')

# Resaltar los puntos mal clasificados con un circulo rojo
plt.scatter(X[feature1][misclassified], X[feature2][misclassified], 
            s=80, facecolors='none', edgecolors='r', linewidths=2)

plt.xlabel(feature1)
plt.ylabel(feature2)
plt.title(f'Fronteras de decision de knn para {feature1} vs {feature2} (k={best_k}, modelo final)')
plt.figtext(0.5, 0.01, 'Nota: Los colores de los puntos muestran las etiquetas verdaderas, los colores de fondo muestran las predicciones del modelo.\nLos circulos rojos indican los puntos mal clasificados.', 
            ha='center', fontsize=9, bbox={"facecolor":"white", "alpha":0.8, "pad":5})


handles, labels = scatter.legend_elements()
legend = plt.legend(handles, np.unique(y), loc="upper right", title="Especies")

plt.tight_layout(rect=[0, 0.05, 1, 1]) #ajustar layout para que entre el subtitulo

# en la figure 6 se ven 5 puntos rojos. Esos son los puntos que fueron mal clasificados.

#    - Iris-setosa (violeta): Claramente separada de las otras especies
#    - Iris-versicolor (celeste): Ubicada en la región central
#    - Iris-virginica (amarillo): Ubicada en la región superior derecha

# el overlapping se da en flores que son iris-virginica pero se clasifican como iris-versicolor

# analisis de regresion logistica con hiperparametros
print("\n--- Analisis de regresion logistica ---")

# Preparar los datos (igual que antes)
X_log = iris_df.drop(['Species', 'Id'], axis=1)  # Features
y_log = iris_df['Species']  # Target variable

# Escalado de las features
X_log_scaled = scaler.fit_transform(X_log)

# Definir hiperparametros a ajustar
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
penalties = ['l1', 'l2', 'elasticnet', None]
solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

# No todas las combinaciones de penalty y solver son validas, asi que crearemos combinaciones validas
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

# Usar GridSearchCV para encontrar los mejores hiperparametros
grid_search = GridSearchCV(
    estimator=LogisticRegression(max_iter=2000, random_state=42),
    param_grid=param_grid,
    cv=10,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Realizar la grid search
print("\nRealizando busqueda de grid para hiperparametros de regresion logistica...")
grid_search.fit(X_log_scaled, y_log)

# Obtener los mejores parametros y score para esos parametros
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"\nMejores hiperparametros: {best_params}")
print(f"Mejor accuracy de cross-validation: {best_score:.4f}")

# Entrenar el modelo final con los mejores parametros
best_log_reg = LogisticRegression(**best_params, random_state=42)
best_log_reg.fit(X_log_scaled, y_log)

# Obtener predicciones del modelo final
log_y_pred = best_log_reg.predict(X_log_scaled)
log_misclassified = log_y_pred != y_log
log_final_accuracy = accuracy_score(y_log, log_y_pred)

print(f"\nAccuracy del modelo final de regresion logistica: {log_final_accuracy:.4f}")
print("\nReporte de clasificacion del modelo final de regresion logistica:")
print(classification_report(y_log, log_y_pred))

print("\nMatriz de confusion de regresion logistica:")
log_conf_matrix = confusion_matrix(y_log, log_y_pred)
print(log_conf_matrix)

# matriz de confusion de regresion logistica
plt.figure(figsize=(8, 6))
sns.heatmap(log_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_log),
            yticklabels=np.unique(y_log))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression Confusion Matrix (Final Model)')

# Analizar importancia de las features a traves de los coeficientes (si es aplicable)
if best_params['penalty'] != 'l1' or best_params['C'] > 0.01:  # Solo si los coeficientes son significativos
    plt.figure(figsize=(10, 6))
    feature_names = X_log.columns
    classes = best_log_reg.classes_
    
    # Crear un DataFrame para almacenar los coeficientes para cada clase
    coef_df = pd.DataFrame(
        best_log_reg.coef_, 
        columns=feature_names,
        index=classes
    )
    
    # Graficar los coeficientes como un mapa de calor
    sns.heatmap(coef_df, annot=True, cmap='coolwarm', fmt='.3f')
    plt.title('Coeficientes de regresion logistica por clase (modelo final)')
    plt.tight_layout()

# Grafico de frontera de decision para el modelo final
plt.figure(figsize=(8, 6))

# Usar las mismas features que en KNN para comparacion
feature1 = 'PetalLengthCm'
feature2 = 'PetalWidthCm'

# Crear una grilla (igual que para KNN)
x_min, x_max = X[feature1].min() - 0.5, X[feature1].max() + 0.5
y_min, y_max = X[feature2].min() - 0.5, X[feature2].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Crear un DataFrame con todas las features establecidas en su media
mesh_df = pd.DataFrame(index=range(xx.size), columns=X.columns)
for col in X.columns:
    mesh_df[col] = X[col].mean()

# Actualizar las dos features que estamos graficando
mesh_df[feature1] = xx.ravel()
mesh_df[feature2] = yy.ravel()

# Obtener predicciones del mejor modelo de regresion logistica
Z_log = best_log_reg.predict(scaler.transform(mesh_df))
Z_log_numeric = np.array([species_to_num[species] for species in Z_log])
Z_log_numeric = Z_log_numeric.reshape(xx.shape)

# Graficar la frontera de decision
plt.contourf(xx, yy, Z_log_numeric, alpha=0.3, cmap='viridis')

# Graficar los puntos de datos, coloreados por su clase actual
scatter = plt.scatter(X[feature1], X[feature2], c=y_numeric, 
           edgecolor='k', s=50, cmap='viridis')

# Resaltar los puntos mal clasificados con un circulo rojo
plt.scatter(X[feature1][log_misclassified], X[feature2][log_misclassified], 
            s=80, facecolors='none', edgecolors='r', linewidths=2)

plt.xlabel(feature1)
plt.ylabel(feature2)
plt.title(f'Fronteras de decision de regresion logistica')
plt.figtext(0.5, 0.01, 'Nota: Los colores de los puntos muestran las etiquetas verdaderas, los colores de fondo muestran las predicciones del modelo.\nLos circulos rojos indican los puntos mal clasificados.', 
            ha='center', fontsize=9, bbox={"facecolor":"white", "alpha":0.8, "pad":5})

handles, labels = scatter.legend_elements()
legend = plt.legend(handles, np.unique(y), loc="upper right", title="Especies")

plt.tight_layout(rect=[0, 0.05, 1, 1])  # Ajustar layout para que entre el subtitulo

# Analizar las clasificaciones incorrectas
print("\nAnalisis de clasificaciones incorrectas (modelos finales):")
print(f"KNN clasificaciones incorrectas: {np.sum(misclassified)}")
print(f"Regresion logistica clasificaciones incorrectas: {np.sum(log_misclassified)}")

# Analizar cuales instancias son mal clasificadas por ambos modelos vs. solo uno
both_misclassified = misclassified & log_misclassified
only_knn_misclassified = misclassified & ~log_misclassified
only_logreg_misclassified = ~misclassified & log_misclassified

print(f"\nInstancias clasificadas incorrectamente por ambos modelos: {np.sum(both_misclassified)}")
print(f"Instancias clasificadas incorrectamente solo por KNN: {np.sum(only_knn_misclassified)}")
print(f"Instancias clasificadas incorrectamente solo por Regresion logistica: {np.sum(only_logreg_misclassified)}")

plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')