import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                            roc_curve, auc, precision_recall_curve, precision_score,
                            recall_score, f1_score, make_scorer)
from scipy.stats import randint

def apply_transformations(df):
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
        
        12: {"new_name": "propiedades", "mapping": {"A124": 4, "A123": 3, "A122": 2, "A121": 1}},
        
        13: {
            "new_name": "edad",
            "transform_func": lambda x: 1 if pd.notna(x) and x < 30 else 1
        },
        
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
    
    columns_to_drop = [col for col in [11, 14] if col in df_transformed.columns]
    if columns_to_drop:
        df_transformed = df_transformed.drop(columns=columns_to_drop)
    
    return df_transformed


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
            "transform_func": lambda x: 1 if pd.notna(x) and x < 30 else 1
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
    
    columns_to_drop = [col for col in [11, 14] if col in df_transformed.columns]
    if columns_to_drop:
        df_transformed = df_transformed.drop(columns=columns_to_drop)
    
    return df_transformed


def calculate_cost(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    
    false_positive_cost = fp * 1
    false_negative_cost = fn * 5
    
    total_cost = false_positive_cost + false_negative_cost
    return -total_cost

df = pd.read_excel('TP 3 Riesgo de credito/Base_Clientes Alemanes.xlsx')

df_transformed = apply_transformations(df)

print("\nTransformed DataFrame Info:")
print(df_transformed.info())

print("\nTransformed DataFrame Description:")
print(df_transformed.describe())

print("Transformed DataFrame:")
print(df_transformed.head())

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

plt.figure(figsize=(10, 6))
rechazo_counts = df_transformed['Rechazo'].value_counts()
colors = ['#3498db', '#e74c3c']
plt.pie(rechazo_counts, labels=['Aceptado (0)', 'Rechazado (1)'], autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Distribución de Aceptación vs Rechazo de Créditos', fontsize=18)
plt.tight_layout()

numeric_cols = df_transformed.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df_transformed.select_dtypes(include=['object']).columns.tolist()

if 'Rechazo' in numeric_cols:
    numeric_cols.remove('Rechazo')

plt.figure(figsize=(14, 12))
correlation_matrix = df_transformed[numeric_cols + ['Rechazo']].corr()
mask = np.zeros_like(correlation_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f', mask=mask)
plt.title('Matriz de Correlación entre Variables Numéricas', fontsize=18)
plt.tight_layout()

plt.figure(figsize=(18, 15))
important_numeric_cols = []

correlations_with_target = abs(correlation_matrix['Rechazo'].drop('Rechazo'))
important_numeric_cols = correlations_with_target.nlargest(6).index.tolist()

for i, col in enumerate(important_numeric_cols):
    plt.subplot(3, 2, i+1)
    sns.histplot(data=df_transformed, x=col, hue='Rechazo', kde=True, element="step", common_norm=False)
    plt.title(f'Distribución de {col} por Estado de Crédito')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')

plt.tight_layout()

if categorical_cols:
    fig, axes = plt.subplots(len(categorical_cols), 1, figsize=(14, 6*len(categorical_cols)))
    
    for i, col in enumerate(categorical_cols):
        ax = axes[i] if len(categorical_cols) > 1 else axes
        crosstab = pd.crosstab(df_transformed[col], df_transformed['Rechazo'], normalize='index') * 100
        crosstab.plot(kind='bar', stacked=True, ax=ax, color=colors)
        ax.set_ylabel('Porcentaje (%)')
        ax.set_title(f'Proporción de Rechazo por {col}')
        ax.legend(['Aceptado (0)', 'Rechazado (1)'])
    
    plt.tight_layout()

plt.figure(figsize=(18, 15))
for i, col in enumerate(important_numeric_cols):
    plt.subplot(3, 2, i+1)
    sns.boxplot(x='Rechazo', y=col, data=df_transformed, hue='Rechazo', palette=['#3498db', '#e74c3c'], legend=False)
    plt.title(f'Distribución de {col} por Estado de Crédito')
    plt.xlabel('Estado de Crédito (0=Aceptado, 1=Rechazado)')
    plt.ylabel(col)

plt.tight_layout()

important_cols = important_numeric_cols[:4] + ['Rechazo']
plt.figure(figsize=(14, 12))
sns.pairplot(df_transformed[important_cols], hue='Rechazo', palette=['#3498db', '#e74c3c'])
plt.suptitle('Relaciones entre Variables Importantes', y=1.02, fontsize=18)
plt.tight_layout()

plt.figure(figsize=(14, 10))
for i, col in enumerate(important_numeric_cols[:4]):
    plt.subplot(2, 2, i+1)
    sns.boxplot(y=df_transformed[col])
    plt.title(f'Outliers en {col}')
    plt.ylabel(col)

plt.tight_layout()

categorical_like_cols = [col for col in df_transformed.columns 
                        if col not in numeric_cols and col != 'Rechazo']

if categorical_like_cols:
    n_cols = len(categorical_like_cols)
    n_rows = (n_cols + 2) // 3
    
    plt.figure(figsize=(16, 4 * n_rows))
    
    for i, col in enumerate(categorical_like_cols):
        plt.subplot(n_rows, 3, i+1)
        sns.countplot(data=df_transformed, x=col, hue='Rechazo', palette=['#3498db', '#e74c3c'])
        plt.title(f'Conteo de {col} por Estado de Crédito')
        plt.xticks(rotation=45)
        plt.legend(['Aceptado (0)', 'Rechazado (1)'])
    
    plt.tight_layout()

df_columnas = apply_transformations_complete(df)

X = df_columnas.drop('Rechazo', axis=1)
y = df_columnas['Rechazo']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

for col in numeric_cols:
    if col in X_train.columns:
        X_train_scaled[col] = scaler.fit_transform(X_train[[col]])
        X_test_scaled[col] = scaler.transform(X_test[[col]])

print("\nRealizando búsqueda de hiperparámetros para regresión logística...")
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
penalties = ['l1', 'l2', 'elasticnet', None]
solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

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

cost_scorer = make_scorer(calculate_cost, greater_is_better=True)

grid_search_accuracy = GridSearchCV(
    estimator=LogisticRegression(max_iter=2000, random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search_accuracy.fit(X_train_scaled, y_train)

log_reg_best_accuracy = LogisticRegression(**grid_search_accuracy.best_params_, random_state=42, max_iter=2000)

log_reg_best_accuracy.fit(X_train_scaled, y_train)

y_pred_logreg_accuracy = log_reg_best_accuracy.predict(X_test_scaled)

y_prob_logreg_accuracy = log_reg_best_accuracy.predict_proba(X_test_scaled)[:, 1]

print("\nMODELO DE REGRESIÓN LOGÍSTICA")
print("Accuracy (Accuracy Optimized):", accuracy_score(y_test, y_pred_logreg_accuracy))
print("\nInforme de Clasificación (Accuracy Optimized):")
print(classification_report(y_test, y_pred_logreg_accuracy))

plt.figure(figsize=(8, 6))
cm_accuracy = confusion_matrix(y_test, y_pred_logreg_accuracy)
sns.heatmap(cm_accuracy, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión - Regresión Logística (Accuracy Optimized)')

plt.figure(figsize=(8, 6))
fpr_accuracy, tpr_accuracy, _ = roc_curve(y_test, y_prob_logreg_accuracy)
roc_auc_accuracy = auc(fpr_accuracy, tpr_accuracy)
plt.plot(fpr_accuracy, tpr_accuracy, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_accuracy:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Regresión Logística (Accuracy Optimized)')
plt.legend(loc="lower right")

plt.figure(figsize=(8, 6))
precision_accuracy, recall_accuracy, _ = precision_recall_curve(y_test, y_prob_logreg_accuracy)
plt.plot(recall_accuracy, precision_accuracy, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall - Regresión Logística (Accuracy Optimized)')

plt.figure(figsize=(12, 8))
feature_importance_accuracy = pd.DataFrame(
    {'Característica': X_train.columns, 'Importancia': np.abs(log_reg_best_accuracy.coef_[0])})
feature_importance_accuracy = feature_importance_accuracy.sort_values('Importancia', ascending=False)
sns.barplot(x='Importancia', y='Característica', data=feature_importance_accuracy.head(15), palette='viridis')
plt.title('Top 15 Características Más Importantes - Regresión Logística (Accuracy Optimized)')
plt.tight_layout()

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': [None] + list(range(5, 30, 5)),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

rf_model = RandomForestClassifier(random_state=42)

print("\nRealizando búsqueda aleatoria de hiperparámetros para Random Forest...")
rf_search_accuracy = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

rf_search_error = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring=cost_scorer,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

rf_search_accuracy.fit(X_train_scaled, y_train)
rf_search_error.fit(X_train_scaled, y_train)

rf_best_accuracy = RandomForestClassifier(**rf_search_accuracy.best_params_, random_state=42)
rf_best_error = RandomForestClassifier(**rf_search_error.best_params_, random_state=42)

rf_best_accuracy.fit(X_train_scaled, y_train)
rf_best_error.fit(X_train_scaled, y_train)

y_pred_rf_accuracy = rf_best_accuracy.predict(X_test_scaled)
y_pred_rf_error = rf_best_error.predict(X_test_scaled)

y_prob_rf_accuracy = rf_best_accuracy.predict_proba(X_test_scaled)[:, 1]
y_prob_rf_error = rf_best_error.predict_proba(X_test_scaled)[:, 1]

print("\nMODELO RANDOM FOREST")
print("Accuracy (Accuracy Optimized):", accuracy_score(y_test, y_pred_rf_accuracy))
print("\nInforme de Clasificación (Accuracy Optimized):")
print(classification_report(y_test, y_pred_rf_accuracy))

plt.figure(figsize=(8, 6))
cm_accuracy = confusion_matrix(y_test, y_pred_rf_accuracy)
cm_error = confusion_matrix(y_test, y_pred_rf_error)
sns.heatmap(cm_accuracy, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión - Random Forest (Accuracy Optimized)')

plt.figure(figsize=(8, 6))
fpr_accuracy, tpr_accuracy, _ = roc_curve(y_test, y_prob_rf_accuracy)
roc_auc_accuracy = auc(fpr_accuracy, tpr_accuracy)
plt.plot(fpr_accuracy, tpr_accuracy, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_accuracy:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Random Forest (Accuracy Optimized)')
plt.legend(loc="lower right")

plt.figure(figsize=(8, 6))
precision_accuracy, recall_accuracy, _ = precision_recall_curve(y_test, y_prob_rf_accuracy)
plt.plot(recall_accuracy, precision_accuracy, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall - Random Forest (Accuracy Optimized)')

plt.figure(figsize=(12, 8))
feature_importance_accuracy = pd.DataFrame(
    {'Característica': X_train.columns, 'Importancia': rf_best_accuracy.feature_importances_})
feature_importance_accuracy = feature_importance_accuracy.sort_values('Importancia', ascending=False)
sns.barplot(x='Importancia', y='Característica', data=feature_importance_accuracy.head(15), palette='viridis')
plt.title('Top 15 Características Más Importantes - Random Forest (Accuracy Optimized)')
plt.tight_layout()

plt.figure(figsize=(10, 8))
fpr_logreg_accuracy, tpr_logreg_accuracy, _ = roc_curve(y_test, y_prob_logreg_accuracy)
roc_auc_logreg_accuracy = auc(fpr_logreg_accuracy, tpr_logreg_accuracy)
plt.plot(fpr_logreg_accuracy, tpr_logreg_accuracy, color='blue', lw=2, 
         label=f'ROC Regresión Logística (AUC = {roc_auc_logreg_accuracy:.2f})')

fpr_rf_accuracy, tpr_rf_accuracy, _ = roc_curve(y_test, y_prob_rf_accuracy)
roc_auc_rf_accuracy = auc(fpr_rf_accuracy, tpr_rf_accuracy)
plt.plot(fpr_rf_accuracy, tpr_rf_accuracy, color='red', lw=2, 
         label=f'ROC Random Forest (AUC = {roc_auc_rf_accuracy:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Comparación de Curvas ROC')
plt.legend(loc="lower right")

plt.figure(figsize=(12, 10))

logreg_importance_accuracy = pd.DataFrame(
    {'Característica': X_train.columns, 'Importancia': np.abs(log_reg_best_accuracy.coef_[0])})
logreg_importance_accuracy = logreg_importance_accuracy.sort_values('Importancia', ascending=False).head(10)

rf_importance_accuracy = pd.DataFrame(
    {'Característica': X_train.columns, 'Importancia': rf_best_accuracy.feature_importances_})
rf_importance_accuracy = rf_importance_accuracy.sort_values('Importancia', ascending=False).head(10)

top_features_accuracy = set(logreg_importance_accuracy['Característica'].tolist() + 
                         rf_importance_accuracy['Característica'].tolist())

combined_importance_accuracy = pd.DataFrame({'Característica': list(top_features_accuracy)})
combined_importance_accuracy['Logistic Regression'] = combined_importance_accuracy['Característica'].map(
    dict(zip(logreg_importance_accuracy['Característica'], logreg_importance_accuracy['Importancia'])))
combined_importance_accuracy['Random Forest'] = combined_importance_accuracy['Característica'].map(
    dict(zip(rf_importance_accuracy['Característica'], rf_importance_accuracy['Importancia'])))

combined_importance_accuracy['Logistic Regression'] = combined_importance_accuracy['Logistic Regression'].fillna(0)
combined_importance_accuracy['Random Forest'] = combined_importance_accuracy['Random Forest'].fillna(0)
combined_importance_accuracy['Logistic Regression'] = combined_importance_accuracy['Logistic Regression'] / combined_importance_accuracy['Logistic Regression'].max()
combined_importance_accuracy['Random Forest'] = combined_importance_accuracy['Random Forest'] / combined_importance_accuracy['Random Forest'].max()

combined_importance_melted_accuracy = pd.melt(combined_importance_accuracy, 
                                     id_vars=['Característica'], 
                                     var_name='Modelo', 
                                     value_name='Importancia Normalizada')

plt.figure(figsize=(14, 10))
sns.barplot(x='Importancia Normalizada', y='Característica', hue='Modelo', 
            data=combined_importance_melted_accuracy, palette=['#3498db', '#e74c3c'])
plt.title('Comparación de Importancia de Características entre Modelos (Accuracy Optimized)')
plt.legend(title='Modelo')
plt.tight_layout()

metrics_accuracy = {
    'Accuracy': [accuracy_score(y_test, y_pred_logreg_accuracy), accuracy_score(y_test, y_pred_rf_accuracy)],
    'Precision (Clase 1)': [
        precision_score(y_test, y_pred_logreg_accuracy), 
        precision_score(y_test, y_pred_rf_accuracy)
    ],
    'Recall (Clase 1)': [
        recall_score(y_test, y_pred_logreg_accuracy), 
        recall_score(y_test, y_pred_rf_accuracy)
    ],
    'F1-Score (Clase 1)': [
        f1_score(y_test, y_pred_logreg_accuracy), 
        f1_score(y_test, y_pred_rf_accuracy)
    ],
    'ROC AUC': [roc_auc_accuracy, roc_auc_rf_accuracy]
}

metrics_df_accuracy = pd.DataFrame(metrics_accuracy, index=['Regresión Logística', 'Random Forest'])

plt.figure(figsize=(12, 6))
sns.heatmap(metrics_df_accuracy, annot=True, cmap='YlGnBu', fmt='.4f', cbar=False)
plt.title('Comparación de Métricas entre Modelos (Accuracy Optimized)')
plt.tight_layout()
models_accuracy_scores = {
    "Logistic Regression (Accuracy)": accuracy_score(y_test, y_pred_logreg_accuracy),
    "Random Forest (Accuracy)": accuracy_score(y_test, y_pred_rf_accuracy)
}
best_model_name_by_accuracy = max(models_accuracy_scores.items(), key=lambda x: x[1])[0]

models_cost_scores = {
    "Random Forest (Cost)": calculate_cost(y_test, y_pred_rf_error)
}
best_model_name_by_cost = max(models_cost_scores.items(), key=lambda x: x[1])[0]

best_predictions_accuracy = y_pred_logreg_accuracy if best_model_name_by_accuracy == "Logistic Regression (Accuracy)" else y_pred_rf_accuracy

best_predictions_cost = y_pred_rf_error if best_model_name_by_cost == "Random Forest (Cost)" else y_pred_rf_error

plt.figure(figsize=(10, 8))
cm_best_accuracy = confusion_matrix(y_test, best_predictions_accuracy)
sns.heatmap(cm_best_accuracy, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title(f'Matriz de Confusión - Mejor Modelo por Accuracy ({best_model_name_by_accuracy})\nAccuracy: {models_accuracy_scores[best_model_name_by_accuracy]:.4f}')
plt.tight_layout()

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
cm_rf_accuracy = confusion_matrix(y_test, y_pred_rf_accuracy)
sns.heatmap(cm_rf_accuracy, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title(f'Random Forest (Accuracy Optimized)\nAccuracy: {accuracy_score(y_test, y_pred_rf_accuracy):.3f}\nCosto: {calculate_cost(y_test, y_pred_rf_accuracy)}')

plt.subplot(1, 2, 2)
cm_rf_error = confusion_matrix(y_test, y_pred_rf_error)
sns.heatmap(cm_rf_error, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title(f'Random Forest (Cost Optimized)\nAccuracy: {accuracy_score(y_test, y_pred_rf_error):.3f}\nCosto: {calculate_cost(y_test, y_pred_rf_error)}')

plt.suptitle('Comparison of Random Forest Models', fontsize=16, y=1.05)
plt.tight_layout()

plt.figtext(0.5, -0.05, 
           'Cost Function: False Positives cost 1 point, False Negatives cost 5 points\nLower cost (closer to zero) is better', 
           ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

print("\n== FINAL MODEL COMPARISON ==")
print("\nRandom Forest:")
print(f"Accuracy Optimized: Accuracy = {accuracy_score(y_test, y_pred_rf_accuracy):.3f}, Cost = {calculate_cost(y_test, y_pred_rf_accuracy)}")
print(f"Cost Optimized:     Accuracy = {accuracy_score(y_test, y_pred_rf_error):.3f}, Cost = {calculate_cost(y_test, y_pred_rf_error)}")

print("\nLogistic Regression:")
print(f"Accuracy Optimized: Accuracy = {accuracy_score(y_test, y_pred_logreg_accuracy):.3f}")

all_costs = [
    calculate_cost(y_test, y_pred_rf_accuracy),
    calculate_cost(y_test, y_pred_rf_error)
]

model_names = [
    "Random Forest (Accuracy Optimized)",
    "Random Forest (Cost Optimized)"
]

best_overall_idx = np.argmax(all_costs)
print(f"\nBest Overall Model: {model_names[best_overall_idx]} with cost = {all_costs[best_overall_idx]}")

plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')