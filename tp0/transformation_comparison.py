import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
from collections import Counter

# Función para encontrar el mejor k para KNN
def find_best_k(X, y, k_range=range(1, 51)):
    cv_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())
    
    # Encontrar el mejor valor de k (el menor k con la mayor accuracy)
    best_k = 0
    best_accuracy = 0
    for i, score in enumerate(cv_scores):
        if score > best_accuracy:
            best_accuracy = score
            best_k = k_range[i]
        elif score == best_accuracy and k_range[i] < best_k:
            best_k = k_range[i]
    
    return best_k, best_accuracy

# Función para evaluar KNN con cross-validation
def evaluate_knn_cv(X, y, k, cv=10):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')
    return scores.mean()

# Función para evaluar regresión logística con cross-validation y GridSearchCV
def evaluate_logreg_cv(X, y, cv=10):
    # Definir hiperparámetros a ajustar
    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    penalties = ['l1', 'l2', 'elasticnet', None]
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

    # No todas las combinaciones de penalty y solver son válidas, así que crearemos combinaciones válidas
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

    # Usar GridSearchCV para encontrar los mejores hiperparámetros
    grid_search = GridSearchCV(
        estimator=LogisticRegression(max_iter=2000, random_state=42),
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,  # Usar todos los núcleos disponibles
        verbose=0
    )

    # Realizar la búsqueda de grid
    grid_search.fit(X, y)
    
    # Devolver la mejor puntuación y los mejores parámetros
    return grid_search.best_score_, grid_search.best_params_

# Cargar el dataset
iris_df = pd.read_csv('tp0/iris.csv')
print("Dataset cargado exitosamente.")

# Crear un diccionario de resultados para almacenar todas las métricas de evaluación
results = {
    'Transformation': [],
    'KNN_Accuracy': [],
    'KNN_Best_K': [],
    'LogReg_Accuracy': [],
    'LogReg_Best_Params': []
}

# 1. Características Originales
print("\nEvaluando Características Originales...")
# Extraer características y objetivo
X_original = iris_df.drop(['Species', 'Id'], axis=1)
y = iris_df['Species']

# Aplicar escalado estándar
scaler_orig = StandardScaler()
X_original_scaled = scaler_orig.fit_transform(X_original)

# KNN con cross-validation
best_k, best_score = find_best_k(X_original_scaled, y)
knn_accuracy = evaluate_knn_cv(X_original_scaled, y, best_k)

# Regresión Logística con cross-validation
logreg_score, logreg_params = evaluate_logreg_cv(X_original_scaled, y)

# Almacenar resultados
results['Transformation'].append('Original')
results['KNN_Accuracy'].append(knn_accuracy)
results['KNN_Best_K'].append(best_k)
results['LogReg_Accuracy'].append(logreg_score)
results['LogReg_Best_Params'].append(logreg_params)

# 2. Escalado MinMax
print("\nEvaluando Escalado MinMax...")
# Aplicar escalado MinMax
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X_original)

# KNN con cross-validation
best_k, best_score = find_best_k(X_minmax, y)
knn_accuracy = evaluate_knn_cv(X_minmax, y, best_k)

# Regresión Logística con cross-validation
logreg_score, logreg_params = evaluate_logreg_cv(X_minmax, y)

# Almacenar resultados
results['Transformation'].append('MinMax Scaling')
results['KNN_Accuracy'].append(knn_accuracy)
results['KNN_Best_K'].append(best_k)
results['LogReg_Accuracy'].append(logreg_score)
results['LogReg_Best_Params'].append(logreg_params)

# 3. Transformación Logarítmica
print("\nEvaluando Transformación Logarítmica...")
# Crear nuevas características
iris_with_log = iris_df.copy()
iris_with_log['LogSepalLength'] = np.log(iris_df['SepalLengthCm'])
iris_with_log['LogSepalWidth'] = np.log(iris_df['SepalWidthCm'])
iris_with_log['LogPetalLength'] = np.log(iris_df['PetalLengthCm'])
iris_with_log['LogPetalWidth'] = np.log(iris_df['PetalWidthCm'])

# Extraer características
X_log = iris_with_log[['LogSepalLength', 'LogSepalWidth', 'LogPetalLength', 'LogPetalWidth']]

# Aplicar escalado estándar
scaler_log = StandardScaler()
X_log_scaled = scaler_log.fit_transform(X_log)

# KNN con cross-validation
best_k, best_score = find_best_k(X_log_scaled, y)
knn_accuracy = evaluate_knn_cv(X_log_scaled, y, best_k)

# Regresión Logística con cross-validation
logreg_score, logreg_params = evaluate_logreg_cv(X_log_scaled, y)

# Almacenar resultados
results['Transformation'].append('Logarithmic')
results['KNN_Accuracy'].append(knn_accuracy)
results['KNN_Best_K'].append(best_k)
results['LogReg_Accuracy'].append(logreg_score)
results['LogReg_Best_Params'].append(logreg_params)

# 4. Promedio de Sépalo y Pétalo
print("\nEvaluando Promedio de Sépalo y Pétalo...")
# Crear nuevas características
iris_with_avg = iris_df.copy()
iris_with_avg['AvgSepalSize'] = (iris_df['SepalLengthCm'] + iris_df['SepalWidthCm']) / 2
iris_with_avg['AvgPetalSize'] = (iris_df['PetalLengthCm'] + iris_df['PetalWidthCm']) / 2

# Extraer características
X_avg = iris_with_avg[['AvgSepalSize', 'AvgPetalSize']]

# Aplicar escalado estándar
scaler_avg = StandardScaler()
X_avg_scaled = scaler_avg.fit_transform(X_avg)

# KNN con cross-validation
best_k, best_score = find_best_k(X_avg_scaled, y)
knn_accuracy = evaluate_knn_cv(X_avg_scaled, y, best_k)

# Regresión Logística con cross-validation
logreg_score, logreg_params = evaluate_logreg_cv(X_avg_scaled, y)

# Almacenar resultados
results['Transformation'].append('Avg Sepal & Petal')
results['KNN_Accuracy'].append(knn_accuracy)
results['KNN_Best_K'].append(best_k)
results['LogReg_Accuracy'].append(logreg_score)
results['LogReg_Best_Params'].append(logreg_params)

# 5. Promedio de Altura y Ancho
print("\nEvaluando Promedio de Altura y Ancho...")
# Crear nuevas características
iris_with_hw = iris_df.copy()
iris_with_hw['AvgHeight'] = (iris_df['SepalLengthCm'] + iris_df['PetalLengthCm']) / 2
iris_with_hw['AvgWidth'] = (iris_df['SepalWidthCm'] + iris_df['PetalWidthCm']) / 2

# Extraer características
X_avg_hw = iris_with_hw[['AvgHeight', 'AvgWidth']]

# Aplicar escalado estándar
scaler_avg_hw = StandardScaler()
X_avg_hw_scaled = scaler_avg_hw.fit_transform(X_avg_hw)

# KNN con cross-validation
best_k, best_score = find_best_k(X_avg_hw_scaled, y)
knn_accuracy = evaluate_knn_cv(X_avg_hw_scaled, y, best_k)

# Regresión Logística con cross-validation
logreg_score, logreg_params = evaluate_logreg_cv(X_avg_hw_scaled, y)

# Almacenar resultados
results['Transformation'].append('Avg Height & Width')
results['KNN_Accuracy'].append(knn_accuracy)
results['KNN_Best_K'].append(best_k)
results['LogReg_Accuracy'].append(logreg_score)
results['LogReg_Best_Params'].append(logreg_params)

# 6. Ratios
print("\nEvaluando Ratios...")
# Crear nuevas características
iris_with_ratios = iris_df.copy()
iris_with_ratios['SepalRatio'] = iris_df['SepalLengthCm'] / iris_df['SepalWidthCm']
iris_with_ratios['PetalRatio'] = iris_df['PetalLengthCm'] / iris_df['PetalWidthCm']
iris_with_ratios['LengthRatio'] = iris_df['SepalLengthCm'] / iris_df['PetalLengthCm']
iris_with_ratios['WidthRatio'] = iris_df['SepalWidthCm'] / iris_df['PetalWidthCm']

# Extraer características
X_ratios = iris_with_ratios[['SepalRatio', 'PetalRatio', 'LengthRatio', 'WidthRatio']]

# Aplicar escalado estándar
scaler_ratios = StandardScaler()
X_ratios_scaled = scaler_ratios.fit_transform(X_ratios)

# KNN con cross-validation
best_k, best_score = find_best_k(X_ratios_scaled, y)
knn_accuracy = evaluate_knn_cv(X_ratios_scaled, y, best_k)

# Regresión Logística con cross-validation
logreg_score, logreg_params = evaluate_logreg_cv(X_ratios_scaled, y)

# Almacenar resultados
results['Transformation'].append('Ratios')
results['KNN_Accuracy'].append(knn_accuracy)
results['KNN_Best_K'].append(best_k)
results['LogReg_Accuracy'].append(logreg_score)
results['LogReg_Best_Params'].append(logreg_params)

# Convertir resultados a DataFrame para facilitar la visualización
results_df = pd.DataFrame(results)
print("\nResumen de Resultados:")
print(results_df)

# Crear una figura para visualizar los resultados
plt.figure(figsize=(16, 12))

# Reemplazar los gráficos de barras con visualizaciones tipo tabla
plt.subplot(1, 2, 1)
# Crear una tabla para la accuracy de KNN
knn_table_data = []
for i, row in results_df.iterrows():
    knn_table_data.append([row['Transformation'], f"{row['KNN_Accuracy']:.4f}"])

# Ordenar por accuracy (descendente)
knn_table_data.sort(key=lambda x: float(x[1]), reverse=True)

# Encontrar el valor máximo de accuracy
max_knn_accuracy = float(knn_table_data[0][1])

# Resaltar todas las transformaciones con la accuracy máxima
for i in range(len(knn_table_data)):
    if float(knn_table_data[i][1]) == max_knn_accuracy:  # Esto coincide con la accuracy más alta
        knn_table_data[i].append('★')  # Agregar una estrella para marcar el mejor
    else:
        knn_table_data[i].append('')

# Agregar encabezados
knn_table_data.insert(0, ['Transformación', 'accuracy', ''])

# Crear la tabla
knn_table = plt.table(cellText=knn_table_data, 
                      loc='center', 
                      cellLoc='center',
                      colWidths=[0.5, 0.3, 0.1])
knn_table.auto_set_font_size(False)
knn_table.set_fontsize(12)
knn_table.scale(1, 1.5)

# Estilizar la tabla - resaltar todas las filas con la accuracy máxima
for i in range(len(knn_table_data)):
    if i > 0:  # Omitir la fila de encabezado
        if knn_table_data[i][2] == '★':  # Esta es una de las mejores filas
            for j in range(3):
                knn_table[(i, j)].set_facecolor('#ff7f0e')
                knn_table[(i, j)].set_text_props(color='white', fontweight='bold')
        else:  # Otras filas de datos
            for j in range(3):
                knn_table[(i, j)].set_facecolor('#1f77b4')
                knn_table[(i, j)].set_text_props(color='white')

plt.title('accuracy de KNN por Transformación', fontsize=14)
plt.axis('off')

# Tabla de accuracy de Regresión Logística
plt.subplot(1, 2, 2)
# Crear una tabla para la accuracy de LogReg
logreg_table_data = []
for i, row in results_df.iterrows():
    logreg_table_data.append([row['Transformation'], f"{row['LogReg_Accuracy']:.4f}"])

# Ordenar por accuracy (descendente)
logreg_table_data.sort(key=lambda x: float(x[1]), reverse=True)

# Encontrar el valor máximo de accuracy
max_logreg_accuracy = float(logreg_table_data[0][1])

# Resaltar todas las transformaciones con la accuracy máxima
for i in range(len(logreg_table_data)):
    if float(logreg_table_data[i][1]) == max_logreg_accuracy:  # Esto coincide con la accuracy más alta
        logreg_table_data[i].append('★')  # Agregar una estrella para marcar el mejor
    else:
        logreg_table_data[i].append('')

# Agregar encabezados
logreg_table_data.insert(0, ['Transformación', 'accuracy', ''])

# Crear la tabla
logreg_table = plt.table(cellText=logreg_table_data, 
                         loc='center', 
                         cellLoc='center',
                         colWidths=[0.5, 0.3, 0.1])
logreg_table.auto_set_font_size(False)
logreg_table.set_fontsize(12)
logreg_table.scale(1, 1.5)

# Estilizar la tabla - resaltar todas las filas con la accuracy máxima
for i in range(len(logreg_table_data)):
    if i > 0:  # Omitir la fila de encabezado
        if logreg_table_data[i][2] == '★':  # Esta es una de las mejores filas
            for j in range(3):
                logreg_table[(i, j)].set_facecolor('#ff7f0e')
                logreg_table[(i, j)].set_text_props(color='white', fontweight='bold')
        else:  # Otras filas de datos
            for j in range(3):
                logreg_table[(i, j)].set_facecolor('#1f77b4')
                logreg_table[(i, j)].set_text_props(color='white')

plt.title('accuracy de Regresión Logística por Transformación', fontsize=14)
plt.axis('off')

plt.tight_layout()
plt.suptitle('Comparación de Métodos de Transformación para Clasificación de Iris', fontsize=16, y=1.02)


# Encontrar la mejor transformación general para cada modelo
best_knn_transform = results_df.loc[results_df['KNN_Accuracy'].idxmax(), 'Transformation']
best_logreg_transform = results_df.loc[results_df['LogReg_Accuracy'].idxmax(), 'Transformation']

print(f"\nMejor transformación para KNN: {best_knn_transform} con accuracy {results_df['KNN_Accuracy'].max():.4f}")
print(f"Mejor transformación para Regresión Logística: {best_logreg_transform} con accuracy {results_df['LogReg_Accuracy'].max():.4f}")

# Enfocarse en la transformación "Avg Sepal & Petal" para evaluación detallada
best_transform = 'Avg Sepal & Petal'
print(f"\nRealizando evaluación detallada para la transformación: {best_transform}")

# Obtener los datos de la transformación seleccionada
X_best = X_avg_scaled
feature_names = X_avg.columns

# Obtener los mejores parámetros para KNN y LogReg con esta transformación
best_k = results_df.loc[results_df['Transformation'] == best_transform, 'KNN_Best_K'].values[0]
best_logreg_params = results_df.loc[results_df['Transformation'] == best_transform, 'LogReg_Best_Params'].values[0]

print(f"\nMejor k para KNN con {best_transform}: {best_k}")
print(f"Mejores parámetros para Regresión Logística con {best_transform}: {best_logreg_params}")

# Entrenar los modelos finales
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_logreg = LogisticRegression(**best_logreg_params, random_state=42, max_iter=2000)

# Evaluar con validación cruzada
knn_cv_scores = cross_val_score(best_knn, X_best, y, cv=10, scoring='accuracy')
logreg_cv_scores = cross_val_score(best_logreg, X_best, y, cv=10, scoring='accuracy')

print(f"\naccuracy de KNN con validación cruzada: {knn_cv_scores.mean():.4f} ± {knn_cv_scores.std():.4f}")
print(f"accuracy de Regresión Logística con validación cruzada: {logreg_cv_scores.mean():.4f} ± {logreg_cv_scores.std():.4f}")

# Entrenar en el conjunto completo para visualización
best_knn.fit(X_best, y)
best_logreg.fit(X_best, y)

# Obtener predicciones
y_pred_knn = best_knn.predict(X_best)
y_pred_logreg = best_logreg.predict(X_best)

# Calcular matrices de confusión
conf_matrix_knn = confusion_matrix(y, y_pred_knn)
conf_matrix_logreg = confusion_matrix(y, y_pred_logreg)

# Calcular accuracy final
knn_accuracy = accuracy_score(y, y_pred_knn)
logreg_accuracy = accuracy_score(y, y_pred_logreg)

print(f"\naccuracy final de KNN: {knn_accuracy:.4f}")
print(f"accuracy final de Regresión Logística: {logreg_accuracy:.4f}")

# Visualizar matrices de confusión
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Matriz de confusión para KNN
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y),
            yticklabels=np.unique(y), ax=ax1)
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')
ax1.set_title(f'Matriz de Confusión de KNN con {best_transform}')

# Matriz de confusión para Regresión Logística
sns.heatmap(conf_matrix_logreg, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y),
            yticklabels=np.unique(y), ax=ax2)
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')
ax2.set_title(f'Matriz de Confusión de Regresión Logística con {best_transform}')

plt.tight_layout()

# Visualizar fronteras de decisión para ambos modelos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Crear una grilla de malla para las dos características
x_min, x_max = X_best[:, 0].min() - 0.5, X_best[:, 0].max() + 0.5
y_min, y_max = X_best[:, 1].min() - 0.5, X_best[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
mesh_X = np.c_[xx.ravel(), yy.ravel()]

# Mapeo de especies a valores numéricos
species_to_num = {species: i for i, species in enumerate(np.unique(y))}
y_numeric = np.array([species_to_num[species] for species in y])

# Frontera de decisión para KNN
Z_knn = best_knn.predict(mesh_X)
Z_knn_numeric = np.array([species_to_num[species] for species in Z_knn])
Z_knn_numeric = Z_knn_numeric.reshape(xx.shape)

ax1.contourf(xx, yy, Z_knn_numeric, alpha=0.3, cmap='viridis')
scatter1 = ax1.scatter(X_best[:, 0], X_best[:, 1], c=y_numeric, 
                      edgecolor='k', s=50, cmap='viridis')
misclassified_knn = y_pred_knn != y
ax1.scatter(X_best[misclassified_knn, 0], X_best[misclassified_knn, 1], 
           s=80, facecolors='none', edgecolors='r', linewidths=2)
ax1.set_xlabel(feature_names[0])
ax1.set_ylabel(feature_names[1])
ax1.set_title(f'Fronteras de Decisión de KNN con {best_transform}')

# Frontera de decisión para Regresión Logística
Z_logreg = best_logreg.predict(mesh_X)
Z_logreg_numeric = np.array([species_to_num[species] for species in Z_logreg])
Z_logreg_numeric = Z_logreg_numeric.reshape(xx.shape)

ax2.contourf(xx, yy, Z_logreg_numeric, alpha=0.3, cmap='viridis')
scatter2 = ax2.scatter(X_best[:, 0], X_best[:, 1], c=y_numeric, 
                      edgecolor='k', s=50, cmap='viridis')
misclassified_logreg = y_pred_logreg != y
ax2.scatter(X_best[misclassified_logreg, 0], X_best[misclassified_logreg, 1], 
           s=80, facecolors='none', edgecolors='r', linewidths=2)
ax2.set_xlabel(feature_names[0])
ax2.set_ylabel(feature_names[1])
ax2.set_title(f'Fronteras de Decisión de Regresión Logística con {best_transform}')

# Mover la leyenda a la esquina superior derecha del gráfico
handles, labels = scatter1.legend_elements()
ax2.legend(handles, np.unique(y), loc="upper right", title="Especies")

fig.suptitle(f'Comparación de Fronteras de Decisión con {best_transform}', fontsize=16, y=0.98)

# Agregar nota explicativa en la parte inferior
plt.figtext(0.5, 0.01, 'Nota: Los colores de los puntos muestran las etiquetas verdaderas, los colores de fondo muestran las predicciones del modelo.\nLos círculos rojos indican los puntos mal clasificados.', 
            ha='center', fontsize=9, bbox={"facecolor":"white", "alpha":0.8, "pad":5})

plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# Reporte de clasificación para ambos modelos
print("\nReporte de Clasificación para KNN:")
print(classification_report(y, y_pred_knn))

print("\nReporte de Clasificación para Regresión Logística:")
print(classification_report(y, y_pred_logreg))

plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')