import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from sklearn.metrics import confusion_matrix

def analyze_quality_distribution(combined_data):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=combined_data, x='quality', hue='type')
    plt.title('Distribución de Puntajes de Calidad por Tipo de Vino')
    plt.xlabel('Puntaje de Calidad')
    plt.ylabel('Cantidad')
    plt.legend(title='Tipo', labels=['Blanco', 'Tinto'])
    plt.tight_layout()
    return plt.gcf()

def perform_rf_feature_sensitivity_analysis(combined_data):
    print("\nIniciando Análisis de Sensibilidad de Características Random Forest...")
    
    X = combined_data.drop(['type', 'quality'], axis=1)
    y = combined_data['quality']
    
    all_features = X.columns.tolist()
    
    best_accuracy = 0
    best_params = None
    best_features = all_features.copy()
    current_features = all_features.copy()
    
    results = []
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=kf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_scaled, y)
    
    initial_best_params = grid_search.best_params_
    initial_accuracy = grid_search.best_score_
    initial_model = grid_search.best_estimator_
    
    best_accuracy = initial_accuracy
    best_params = initial_best_params.copy()
    best_model = initial_model
    best_scaler = scaler
    
    results.append({
        'Características Eliminadas': 'Ninguna',
        'Características Usadas': len(current_features),
        'Mejores Parámetros': str(initial_best_params),
        'Precisión': initial_accuracy
    })
    
    print(f"Modelo inicial (todas las {len(current_features)} características): Precisión = {initial_accuracy:.4f}")
    
    feature_importance = dict(zip(all_features, initial_model.feature_importances_))
    
    features_by_importance = sorted(feature_importance.items(), key=lambda x: x[1])
    features_by_importance = [f[0] for f in features_by_importance]
    
    for feature_to_remove in features_by_importance:
        if feature_to_remove not in current_features:
            continue
            
        current_features.remove(feature_to_remove)
        
        if len(current_features) == 0:
            break
            
        X_reduced = X[current_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reduced)
        
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=kf,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_scaled, y)
        
        current_best_params = grid_search.best_params_
        current_accuracy = grid_search.best_score_
        current_model = grid_search.best_estimator_
        
        results.append({
            'Características Eliminadas': feature_to_remove,
            'Características Usadas': len(current_features),
            'Mejores Parámetros': str(current_best_params),
            'Precisión': current_accuracy
        })
        
        print(f"Eliminada '{feature_to_remove}': Precisión = {current_accuracy:.4f}")
        
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_params = current_best_params.copy()
            best_features = current_features.copy()
            best_model = current_model
            best_scaler = scaler
            print(f"  --> ¡Nuevo mejor modelo encontrado!")
    
    results_df = pd.DataFrame(results)
    
    print("\nResumen del Mejor Modelo:")
    print(f"Características: {best_features}")
    print(f"Número de características: {len(best_features)}")
    print(f"Mejores parámetros: {best_params}")
    print(f"Precisión de validación cruzada: {best_accuracy:.4f}")
    
    plt.figure(figsize=(15, 8))
    
    x_labels = ['Todas las características']
    x_labels.extend([row['Características Eliminadas'] for i, row in results_df[1:].iterrows()])
    
    plt.plot(range(len(results)), results_df['Precisión'], 'o-', linewidth=2)
    
    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha='right')
    
    plt.xlabel('Característica Eliminada', fontsize=12)
    plt.ylabel('Precisión', fontsize=12)
    plt.title('Precisión del Modelo vs. Características Eliminadas', fontsize=16)
    plt.grid(True)
    
    best_idx = results_df['Precisión'].idxmax()
    plt.plot(best_idx, results_df.loc[best_idx, 'Precisión'], 'ro', markersize=10)
    
    mejor_caracteristica = x_labels[best_idx]
    plt.annotate(f'Mejor Modelo\n(después de eliminar\n{mejor_caracteristica})',
                xy=(best_idx, results_df.loc[best_idx, 'Precisión']),
                xytext=(10, 20),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->'),
                fontsize=10,
                color='red',
                bbox=dict(facecolor='white', edgecolor='red', alpha=0.8))
    
    plt.margins(x=0.1)
    plt.subplots_adjust(bottom=0.2)
    
    sensitivity_fig = plt.gcf()
    
    conf_matrix = create_confusion_matrix_for_model(
        best_model, X[best_features], y, best_scaler,
        "Random Forest Feature Selection Best Model Confusion Matrix"
    )
    
    relaxed_conf_matrix = create_relaxed_confusion_matrix_for_model(
        best_model, X[best_features], y, best_scaler,
        "Random Forest Feature Selection Best Model Confusion Matrix (±1 Tolerance)"
    )
    
    return best_model, best_scaler, best_features, best_params, best_accuracy, [sensitivity_fig, conf_matrix, relaxed_conf_matrix]

def train_rf_quality_model(combined_data):
    X = combined_data.drop(['type', 'quality'], axis=1)
    y = combined_data['quality']
    
    feature_names = X.columns.tolist()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=kf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_scaled, y)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_accuracy = grid_search.best_score_
    
    print("\nResultados de la Búsqueda de Cuadrícula Random Forest:")
    print(f"Mejores Parámetros: {best_params}")
    print(f"Mejor Precisión de Validación Cruzada: {best_accuracy:.4f}")
    
    plt.figure(figsize=(12, 8))
    
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    importance_df = pd.DataFrame({
        'Característica': [feature_names[i] for i in indices],
        'Importancia': [importances[i] for i in indices]
    })
    
    importance_df['Significativa'] = importance_df['Importancia'] > np.mean(importances)
    
    ax = sns.barplot(x='Importancia', y='Característica', data=importance_df, color='lightblue')
    
    for i, row in enumerate(importance_df.itertuples()):
        value_text = f"{row.Importancia:.3f}"
        ax.text(row.Importancia + 0.005, i, value_text, fontsize=10, va='center')
    
    plt.title('Importancia de Características para Predicción de Calidad del Vino', fontsize=16)
    plt.xlabel('Importancia', fontsize=12)
    plt.ylabel('Característica', fontsize=12)
    plt.tight_layout()
    
    return best_model, scaler, best_params, best_accuracy, plt.gcf(), feature_names

def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    white_wine_path = os.path.join(script_dir, 'winequality-white.csv')
    red_wine_path = os.path.join(script_dir, 'winequality-red.csv')
    
    white_wine_data = pd.read_csv(white_wine_path, sep=';')
    red_wine_data = pd.read_csv(red_wine_path, sep=';')

    white_wine_data['type'] = 'white'
    red_wine_data['type'] = 'red'

    combined_data = pd.concat([white_wine_data, red_wine_data], axis=0)
    
    combined_data = combined_data.reset_index(drop=True)
    
    print("\nResumen del Conjunto de Datos:")
    print(f"Número total de muestras: {len(combined_data)}")
    print(f"Número de vinos blancos: {len(white_wine_data)}")
    print(f"Número de vinos tintos: {len(red_wine_data)}")
    print(f"\nRango de puntajes de calidad: {combined_data['quality'].min()} a {combined_data['quality'].max()}")
    
    return combined_data

def create_confusion_matrix_for_model(model, X, y, scaler, title):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    y_pred = np.zeros_like(y)
    
    X_scaled = scaler.transform(X)
    
    for train_idx, test_idx in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        
        fold_model = model.__class__(**model.get_params())
        
        fold_model.fit(X_train, y_train)
        
        y_pred[test_idx] = fold_model.predict(X_test)
    
    return plot_confusion_matrix(y, y_pred, title)

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    
    quality_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    
    sns.heatmap(cm[::-1], annot=True, fmt='d', cmap='Blues',
                xticklabels=quality_labels,
                yticklabels=quality_labels[::-1])
    
    plt.title(title)
    plt.ylabel('True Quality')
    plt.xlabel('Predicted Quality')
    
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    plt.figtext(0.99, 0.01, f'Accuracy: {accuracy:.3f}',
                horizontalalignment='right', fontsize=10)
    
    return plt.gcf()

def calculate_relaxed_accuracy(y_true, y_pred, tolerance=1):
    correct = 0
    total = len(y_true)
    
    for true, pred in zip(y_true, y_pred):
        if abs(true - pred) <= tolerance:
            correct += 1
            
    return correct / total

def create_relaxed_confusion_matrix_for_model(model, X, y, scaler, title, tolerance=1):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    y_pred = np.zeros_like(y)
    
    X_scaled = scaler.transform(X)
    
    for train_idx, test_idx in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        
        fold_model = model.__class__(**model.get_params())
        
        fold_model.fit(X_train, y_train)
        
        y_pred[test_idx] = fold_model.predict(X_test)
    
    return plot_relaxed_confusion_matrix(y, y_pred, title, tolerance)

def plot_relaxed_confusion_matrix(y_true, y_pred, title="Confusion Matrix (±1 Tolerance)", tolerance=1):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    
    quality_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    
    mask = np.zeros_like(cm, dtype=bool)
    for i in range(len(quality_labels)):
        for j in range(max(0, i-tolerance), min(len(quality_labels), i+tolerance+1)):
            mask[i][j] = True
    
    cm = cm[::-1]
    mask = mask[::-1]
    
    colors = ['#c6dbef', '#4292c6']
    cmap = sns.color_palette(colors, as_cmap=True)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=quality_labels,
                yticklabels=quality_labels[::-1],
                mask=~mask)
    
    plt.title(title)
    plt.ylabel('True Quality')
    plt.xlabel('Predicted Quality')
    
    standard_accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    relaxed_accuracy = calculate_relaxed_accuracy(y_true, y_pred, tolerance)
    
    plt.figtext(0.99, 0.05, f'Standard Accuracy: {standard_accuracy:.3f}',
                horizontalalignment='right', fontsize=10)
    plt.figtext(0.99, 0.01, f'Accuracy (±{tolerance}): {relaxed_accuracy:.3f}',
                horizontalalignment='right', fontsize=10)
    
    return plt.gcf()

def perform_anova_feature_sensitivity_analysis(combined_data):
    print("\nIniciando Análisis de Sensibilidad de Características ANOVA...")
    
    anova_results = analyze_anova(combined_data)
    
    X = combined_data.drop(['type', 'quality'], axis=1)
    y = combined_data['quality']
    
    all_features = X.columns.tolist()
    
    features_by_importance = anova_results[0]['Variable'].tolist()
    features_by_importance.reverse()
    
    best_accuracy = 0
    best_params = None
    best_features = all_features.copy()
    current_features = all_features.copy()
    
    results = []
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=kf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_scaled, y)
    
    initial_best_params = grid_search.best_params_
    initial_accuracy = grid_search.best_score_
    initial_model = grid_search.best_estimator_
    
    best_accuracy = initial_accuracy
    best_params = initial_best_params.copy()
    best_model = initial_model
    best_scaler = scaler
    
    results.append({
        'Características Eliminadas': 'Ninguna',
        'Características Usadas': len(current_features),
        'Mejores Parámetros': str(initial_best_params),
        'Precisión': initial_accuracy
    })
    
    print(f"Modelo inicial (todas las {len(current_features)} características): Precisión = {initial_accuracy:.4f}")
    
    for feature_to_remove in features_by_importance:
        if feature_to_remove not in current_features:
            continue
            
        current_features.remove(feature_to_remove)
        
        if len(current_features) == 0:
            break
            
        X_reduced = X[current_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reduced)
        
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=kf,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_scaled, y)
        
        current_best_params = grid_search.best_params_
        current_accuracy = grid_search.best_score_
        current_model = grid_search.best_estimator_
        
        results.append({
            'Características Eliminadas': feature_to_remove,
            'Características Usadas': len(current_features),
            'Mejores Parámetros': str(current_best_params),
            'Precisión': current_accuracy
        })
        
        print(f"Eliminada '{feature_to_remove}': Precisión = {current_accuracy:.4f}")
        
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_params = current_best_params.copy()
            best_features = current_features.copy()
            best_model = current_model
            best_scaler = scaler
            print(f"  --> ¡Nuevo mejor modelo encontrado!")
    
    results_df = pd.DataFrame(results)
    
    print("\nResumen del Mejor Modelo (ANOVA):")
    print(f"Características: {best_features}")
    print(f"Número de características: {len(best_features)}")
    print(f"Mejores parámetros: {best_params}")
    print(f"Precisión de validación cruzada: {best_accuracy:.4f}")
    
    plt.figure(figsize=(15, 8))
    
    x_labels = ['Todas las características']
    x_labels.extend([row['Características Eliminadas'] for i, row in results_df[1:].iterrows()])
    
    plt.plot(range(len(results)), results_df['Precisión'], 'o-', linewidth=2)
    
    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha='right')
    
    plt.xlabel('Característica Eliminada', fontsize=12)
    plt.ylabel('Precisión', fontsize=12)
    plt.title('Precisión del Modelo vs. Características Eliminadas (ANOVA)', fontsize=16)
    plt.grid(True)
    
    best_idx = results_df['Precisión'].idxmax()
    plt.plot(best_idx, results_df.loc[best_idx, 'Precisión'], 'ro', markersize=10)
    
    plt.tight_layout()
    sensitivity_fig = plt.gcf()
    
    conf_matrix = create_confusion_matrix_for_model(
        best_model, X[best_features], y, best_scaler,
        "Random Forest ANOVA Feature Selection Best Model Confusion Matrix"
    )
    
    relaxed_conf_matrix = create_relaxed_confusion_matrix_for_model(
        best_model, X[best_features], y, best_scaler,
        "Random Forest ANOVA Feature Selection Best Model Confusion Matrix (±1 Tolerance)"
    )
    
    return best_model, best_scaler, best_features, best_params, best_accuracy, [sensitivity_fig, conf_matrix, relaxed_conf_matrix]

def analyze_anova(combined_data):
    numeric_columns = combined_data.drop(['type', 'quality'], axis=1).columns
    
    data_for_analysis = combined_data.copy()
    
    scaler = StandardScaler()
    data_for_analysis[numeric_columns] = scaler.fit_transform(data_for_analysis[numeric_columns])
    
    results = []
    
    for column in numeric_columns:
        groups = [group[column].values for name, group in data_for_analysis.groupby('quality')]
        
        f_val, p_val = stats.f_oneway(*groups)
        
        all_values = data_for_analysis[column]
        grand_mean = all_values.mean()
        total_ss = ((all_values - grand_mean)**2).sum()
        
        between_ss = sum(len(group) * ((group.mean() - grand_mean)**2) 
                        for group in groups)
        
        eta_squared = between_ss / total_ss if total_ss != 0 else 0
        
        results.append({
            'Variable': column,
            'F_statistic': f_val,
            'P_value': p_val,
            'Eta_squared': eta_squared,
            'Significant': p_val < 0.05
        })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('Eta_squared', ascending=False)
    
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Eta_squared', y='Variable', hue='Significant', 
                    data=result_df, palette=['lightblue', 'darkblue'])
    
    for i, row in enumerate(result_df.itertuples()):
        value_text = f"{getattr(row, 'Eta_squared'):.3f}"
        ax.text(getattr(row, 'Eta_squared') + 0.01, i, value_text, va='center')
    
    plt.title('Importancia de Variables para la Calidad del Vino (ANOVA)', fontsize=16)
    plt.xlabel('Tamaño del Efecto (Eta-cuadrado)', fontsize=12)
    plt.ylabel('Propiedades del Vino', fontsize=12)
    plt.tight_layout()
    
    return result_df, plt.gcf()

if __name__ == "__main__":
    print('Cargando y combinando conjuntos de datos...')
    combined_data = load_data()
    
    print('\nAnalizando distribución de calidad...')
    quality_dist_fig = analyze_quality_distribution(combined_data)
    
    print('\nRealizando análisis ANOVA...')
    anova_results, anova_fig = analyze_anova(combined_data)
    
    print('\nEntrenando modelo Random Forest completo...')
    rf_model, scaler, best_params, best_accuracy, importance_fig, feature_names = train_rf_quality_model(combined_data)
    
    print('\nCreando matriz de confusión para el modelo completo...')
    X_full = combined_data.drop(['type', 'quality'], axis=1)
    y = combined_data['quality']
    conf_matrix_full = create_confusion_matrix_for_model(
        rf_model, X_full, y, scaler, 
        "Random Forest Full Model Confusion Matrix"
    )
    
    print('\nCreando matriz de confusión relajada para el modelo completo...')
    relaxed_conf_matrix_full = create_relaxed_confusion_matrix_for_model(
        rf_model, X_full, y, scaler,
        "Random Forest Full Model Confusion Matrix (±1 Tolerance)"
    )
    
    print('\nRealizando análisis de sensibilidad basado en importancia de características RF...')
    rf_best_model, rf_best_scaler, rf_best_features, rf_best_params, rf_best_accuracy, rf_sensitivity_figs = perform_rf_feature_sensitivity_analysis(combined_data)
    
    print('\nRealizando análisis de sensibilidad basado en ANOVA...')
    anova_best_model, anova_best_scaler, anova_best_features, anova_best_params, anova_best_accuracy, anova_sensitivity_figs = perform_anova_feature_sensitivity_analysis(combined_data)
    
    print("\nEvaluando todos los modelos con tolerancia ±1:")
    
    X = combined_data.drop(['type', 'quality'], axis=1)
    y = combined_data['quality']
    y_pred_full = np.zeros_like(y)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        fold_model = RandomForestClassifier(**rf_model.get_params())
        fold_model.fit(scaler.transform(X_train), y_train)
        y_pred_full[test_idx] = fold_model.predict(scaler.transform(X_test))
    
    full_relaxed_acc = calculate_relaxed_accuracy(y, y_pred_full)
    print(f"Modelo RF Completo Precisión Relajada: {full_relaxed_acc:.3f}")
    
    X_rf = combined_data[rf_best_features]
    y_pred_rf = np.zeros_like(y)
    
    for train_idx, test_idx in kf.split(X_rf):
        X_train, X_test = X_rf.iloc[train_idx], X_rf.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        fold_model = RandomForestClassifier(**rf_best_model.get_params())
        fold_model.fit(rf_best_scaler.transform(X_train), y_train)
        y_pred_rf[test_idx] = fold_model.predict(rf_best_scaler.transform(X_test))
    
    rf_relaxed_acc = calculate_relaxed_accuracy(y, y_pred_rf)
    print(f"Modelo RF con Selección de Características RF Precisión Relajada: {rf_relaxed_acc:.3f}")
    
    X_anova = combined_data[anova_best_features]
    y_pred_anova = np.zeros_like(y)
    
    for train_idx, test_idx in kf.split(X_anova):
        X_train, X_test = X_anova.iloc[train_idx], X_anova.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        fold_model = RandomForestClassifier(**anova_best_model.get_params())
        fold_model.fit(anova_best_scaler.transform(X_train), y_train)
        y_pred_anova[test_idx] = fold_model.predict(anova_best_scaler.transform(X_test))
    
    anova_relaxed_acc = calculate_relaxed_accuracy(y, y_pred_anova)
    print(f"Modelo RF con Selección de Características ANOVA Precisión Relajada: {anova_relaxed_acc:.3f}")
    
    plt.show(block=False)
    input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
    print("Programa finalizado. Cerrando figuras...")
    plt.close('all')
