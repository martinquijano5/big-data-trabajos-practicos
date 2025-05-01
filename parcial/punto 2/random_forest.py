import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report)
import os

# Crear directorio para gráficos si no existe
output_dir = 'graficos/modelo_random_forest'
os.makedirs(output_dir, exist_ok=True)

# Cargar datos
print("Cargando datos...")
df = pd.read_csv('punto 2/diabetes_binary_health_indicators_BRFSS2015_transformed.csv')

# Separar features y target
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']

# División train/test (igual que en regresión logística)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Definir espacio de hiperparámetros para RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200],
    'max_depth': [20, 30, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt']
}

# Inicializar modelo base
rf_base = RandomForestClassifier(random_state=42)

# Realizar búsqueda aleatoria de hiperparámetros
print("\nRealizando búsqueda de hiperparámetros...")
rf_random = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    random_state=42,
    n_jobs=-1,
    scoring='roc_auc'
)

# Entrenar modelo con búsqueda de hiperparámetros
rf_random.fit(X_train, y_train)

# Obtener mejor modelo
best_rf = rf_random.best_estimator_

# Realizar predicciones
y_pred = best_rf.predict(X_test)
y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

# Calcular métricas
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1': f1_score(y_test, y_pred),
    'AUC': roc_auc_score(y_test, y_pred_proba)
}

# Imprimir resultados
print("\nMejores hiperparámetros encontrados:")
print(rf_random.best_params_)

print("\nMétricas del modelo:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nMatriz de Confusión:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Graficar importancia de variables (top 10)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Variables más Importantes (Random Forest)')
plt.tight_layout()
plt.savefig(f'{output_dir}/feature_importance.png')
plt.close()

# Graficar matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión (Random Forest)')
plt.ylabel('Real')
plt.xlabel('Predicho')
plt.tight_layout()
plt.savefig(f'{output_dir}/confusion_matrix.png')
plt.close()
