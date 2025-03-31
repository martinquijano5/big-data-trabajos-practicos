import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from sklearn.metrics import confusion_matrix

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def analyze_anova(combined_data):
    numeric_columns = combined_data.drop('quality', axis=1).columns
    
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

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    
    quality_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    
    sns.heatmap(cm[::-1], annot=True, fmt='d', cmap='Blues',
                xticklabels=quality_labels,
                yticklabels=quality_labels[::-1])  # Reverse y-axis labels
    
    plt.title(title)
    plt.ylabel('True Quality')
    plt.xlabel('Predicted Quality')
    
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    plt.figtext(0.99, 0.01, f'Accuracy: {accuracy:.3f}',
                horizontalalignment='right', fontsize=10)
    
    return plt.gcf()

def create_confusion_matrix_for_model(model, X, y, scaler, title):

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    y_pred = np.zeros_like(y)
    
    X_scaled = scaler.transform(X)
    
    for train_idx, test_idx in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        
        if isinstance(model, KNeighborsClassifier):
            fold_model = KNeighborsClassifier(n_neighbors=model.n_neighbors)
        else:
            fold_model = model.__class__(**model.get_params())
        
        fold_model.fit(X_train, y_train)
        
        y_pred[test_idx] = fold_model.predict(X_test)
    
    return plot_confusion_matrix(y, y_pred, title)

def train_knn_model(combined_data):
    X = combined_data.drop('quality', axis=1)
    y = combined_data['quality']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    mean_accuracies = []
    
    for k in range(1, 51):
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_scaled, y, cv=kf, scoring='accuracy')
        mean_accuracies.append(scores.mean())
    
    best_k = range(1, 51)[np.argmax(mean_accuracies)]
    best_accuracy = max(mean_accuracies)
    
    final_model = KNeighborsClassifier(n_neighbors=best_k)
    final_model.fit(X_scaled, y)
    
    conf_matrix = create_confusion_matrix_for_model(
        final_model, X, y, scaler, 
        "Initial KNN Model Confusion Matrix"
    )
    
    return final_model, scaler, best_k, best_accuracy, [conf_matrix]

def perform_knn_feature_sensitivity_analysis(combined_data):

    print("\nIniciando análisis de sensibilidad KNN...")
    
    X = combined_data.drop('quality', axis=1)
    y = combined_data['quality']
    
    all_features = X.columns.tolist()
    
    best_accuracy = 0
    best_k = 0
    best_features = all_features.copy()
    current_features = all_features.copy()
    
    results = []
    feature_importance = {}
    
    initial_model, initial_scaler, initial_best_k, initial_accuracy = train_model_and_evaluate(X, y)
    
    best_accuracy = initial_accuracy
    best_k = initial_best_k
    
    results.append({
        'Features_Removed': 'None',
        'Features_Used': len(current_features),
        'Best_k': initial_best_k,
        'Accuracy': initial_accuracy
    })
    
    print(f"Modelo inicial (todas las características {len(current_features)}): Accuracy = {initial_accuracy:.4f}, k = {initial_best_k}")
    
    def calculate_feature_importance(X, y, features):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        knn = KNeighborsClassifier(n_neighbors=initial_best_k)
        knn.fit(X_scaled, y)
        
        base_accuracy = knn.score(X_scaled, y)
        
        importance = {}
        for i, feature in enumerate(features):
            X_permuted = X_scaled.copy()
            np.random.shuffle(X_permuted[:, i])
            permuted_accuracy = knn.score(X_permuted, y)
            importance[feature] = base_accuracy - permuted_accuracy
        
        return importance
    
    feature_importance = calculate_feature_importance(X, y, current_features)
    
    features_by_importance = sorted(feature_importance.items(), key=lambda x: x[1])
    features_by_importance = [f[0] for f in features_by_importance]
    
    initial_importance_df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    })
    initial_importance_df = initial_importance_df.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Importance', y='Feature', data=initial_importance_df, color='lightblue')
    
    for i, row in enumerate(initial_importance_df.itertuples()):
        value_text = f"{row.Importance:.3f}"
        ax.text(row.Importance + 0.001, i, value_text, fontsize=10, va='center')
    
    plt.title('Importancia de Variables para la Calidad del Vino (Permutación KNN)', fontsize=16)
    plt.xlabel('Importancia (Caída en Precisión al Permutar)', fontsize=12)
    plt.ylabel('Propiedades del Vino', fontsize=12)
    plt.tight_layout()
    
    initial_importance_fig = plt.gcf()
    
    best_model = initial_model
    best_scaler = initial_scaler
    best_model_importance = feature_importance.copy()
    
    for feature_to_remove in features_by_importance:
        if feature_to_remove not in current_features:
            continue
            
        current_features.remove(feature_to_remove)
        
        if len(current_features) == 0:
            break
            
        X_reduced = X[current_features]
        current_model, current_scaler, current_k, current_accuracy = train_model_and_evaluate(X_reduced, y)
        
        results.append({
            'Features_Removed': feature_to_remove,
            'Features_Used': len(current_features),
            'Best_k': current_k,
            'Accuracy': current_accuracy
        })
        
        print(f"Eliminada '{feature_to_remove}': Accuracy = {current_accuracy:.4f}, k = {current_k}")
        
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_k = current_k
            best_features = current_features.copy()
            best_model = current_model
            best_scaler = current_scaler
            best_model_importance = calculate_feature_importance(X_reduced, y, current_features)
            print(f"  --> ¡Nuevo mejor modelo encontrado!")
    
    results_df = pd.DataFrame(results)
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(results)), results_df['Accuracy'], 'o-', linewidth=2)
    plt.xlabel('Iteración (Características Eliminadas)', fontsize=12)
    plt.ylabel('Precisión', fontsize=12)
    plt.title('Precisión del Modelo KNN vs. Características Eliminadas', fontsize=16)
    plt.grid(True)
    
    for i, row in enumerate(results_df.itertuples()):
        if i > 0:
            plt.annotate(row.Features_Removed,
                        xy=(i, row.Accuracy),
                        xytext=(5, 0),
                        textcoords='offset points',
                        rotation=45,
                        fontsize=8)
    
    best_idx = results_df['Accuracy'].idxmax()
    plt.plot(best_idx, results_df.loc[best_idx, 'Accuracy'], 'ro', markersize=10)
    plt.annotate('Mejor Modelo',
                xy=(best_idx, results_df.loc[best_idx, 'Accuracy']),
                xytext=(10, -20),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->'),
                fontsize=12)
    
    plt.tight_layout()
    sensitivity_fig = plt.gcf()
    
    best_conf_matrix = create_confusion_matrix_for_model(
        best_model, X[best_features], y, best_scaler,
        "KNN Feature Selection Best Model Confusion Matrix"
    )
    
    return best_model, best_scaler, best_features, best_k, best_accuracy, [initial_importance_fig, sensitivity_fig, best_conf_matrix]

def train_model_and_evaluate(X, y, k_range=range(1, 51)):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    mean_accuracies = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_scaled, y, cv=kf, scoring='accuracy')
        mean_accuracies.append(scores.mean())
    
    best_k = k_range[np.argmax(mean_accuracies)]
    best_accuracy = max(mean_accuracies)
    
    final_model = KNeighborsClassifier(n_neighbors=best_k)
    final_model.fit(X_scaled, y)
    
    return final_model, scaler, best_k, best_accuracy

def plot_model_k_accuracy(combined_data, best_features):

    X = combined_data[best_features]
    y = combined_data['quality']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    k_values = range(1, 51)
    mean_accuracies = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_scaled, y, cv=kf, scoring='accuracy')
        mean_accuracies.append(scores.mean())
    
    best_k = k_values[np.argmax(mean_accuracies)]
    best_accuracy = max(mean_accuracies)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, mean_accuracies, 'o-')
    plt.axvline(x=best_k, color='r', linestyle='--')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title(f'KNN Model ({len(best_features)} features): Accuracy vs k\nBest k = {best_k}, Accuracy = {best_accuracy:.4f}')
    plt.grid(True)
    
    return plt.gcf(), best_k, best_accuracy

def perform_anova_feature_sensitivity_analysis(combined_data, anova_results):
    print("\nStarting ANOVA-based Feature Sensitivity Analysis...")
    
    X = combined_data.drop('quality', axis=1)
    y = combined_data['quality']
    
    all_features = X.columns.tolist()
    
    features_by_importance = anova_results['Variable'].tolist()
    features_by_importance.reverse()  # Reverse to get least important first
    
    best_accuracy = 0
    best_k = 0
    best_features = all_features.copy()
    current_features = all_features.copy()
    
    results = []
    
    initial_model, initial_scaler, initial_best_k, initial_accuracy = train_model_and_evaluate(X, y)
    
    best_accuracy = initial_accuracy
    best_k = initial_best_k
    best_model = initial_model
    best_scaler = initial_scaler
    
    results.append({
        'Features_Removed': 'None',
        'Features_Used': len(current_features),
        'Best_k': initial_best_k,
        'Accuracy': initial_accuracy
    })
    
    print(f"Initial model (all {len(current_features)} features): Accuracy = {initial_accuracy:.4f}, k = {initial_best_k}")
    
    for feature_to_remove in features_by_importance:
        if feature_to_remove not in current_features:
            continue
            
        current_features.remove(feature_to_remove)
        
        if len(current_features) == 0:
            break
            
        X_reduced = X[current_features]
        current_model, current_scaler, current_k, current_accuracy = train_model_and_evaluate(X_reduced, y)
        
        results.append({
            'Features_Removed': feature_to_remove,
            'Features_Used': len(current_features),
            'Best_k': current_k,
            'Accuracy': current_accuracy
        })
        
        print(f"Removed '{feature_to_remove}': Accuracy = {current_accuracy:.4f}, k = {current_k}")
        
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_k = current_k
            best_features = current_features.copy()
            best_model = current_model
            best_scaler = current_scaler
            print(f"  --> New best model found!")
    
    results_df = pd.DataFrame(results)
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(results)), results_df['Accuracy'], 'o-', linewidth=2)
    plt.xlabel('Iteration (Features Removed)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('KNN Model Accuracy vs. Features Removed (Using ANOVA Feature Importance)', fontsize=16)
    plt.grid(True)
    
    for i, row in enumerate(results_df.itertuples()):
        if i > 0:
            plt.annotate(row.Features_Removed,
                        xy=(i, row.Accuracy),
                        xytext=(5, 0),
                        textcoords='offset points',
                        rotation=45,
                        fontsize=8)
    
    best_idx = results_df['Accuracy'].idxmax()
    plt.plot(best_idx, results_df.loc[best_idx, 'Accuracy'], 'ro', markersize=10)
    plt.annotate('Best Model',
                xy=(best_idx, results_df.loc[best_idx, 'Accuracy']),
                xytext=(10, -20),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->'),
                fontsize=12)
    
    plt.tight_layout()
    sensitivity_fig = plt.gcf()
    
    anova_conf_matrix = create_confusion_matrix_for_model(
        best_model, X[best_features], y, best_scaler,
        "ANOVA Feature Selection Best Model Confusion Matrix"
    )
    
    return best_model, best_scaler, best_features, best_k, best_accuracy, [sensitivity_fig, anova_conf_matrix]

def train_simple_knn_model(combined_data):

    X = combined_data[['volatile acidity', 'total sulfur dioxide']]
    y = combined_data['quality']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    k_values = range(1, 51)
    mean_accuracies = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_scaled, y, cv=kf, scoring='accuracy')
        mean_accuracies.append(scores.mean())
    
    best_k = k_values[np.argmax(mean_accuracies)]
    best_accuracy = max(mean_accuracies)
    
    final_model = KNeighborsClassifier(n_neighbors=best_k)
    final_model.fit(X_scaled, y)
    
    simple_conf_matrix = create_confusion_matrix_for_model(
        final_model, X, y, scaler,
        "Simple KNN Model (2 Features) Confusion Matrix"
    )
    
    return final_model, scaler, best_k, best_accuracy, simple_conf_matrix

script_dir = os.path.dirname(os.path.abspath(__file__))
white_wine_path = os.path.join(script_dir, 'winequality-white.csv')
red_wine_path = os.path.join(script_dir, 'winequality-red.csv')

white_wine_data = pd.read_csv(white_wine_path, sep=';')
red_wine_data = pd.read_csv(red_wine_path, sep=';')

combined_data = pd.concat([white_wine_data, red_wine_data], axis=0)

print("Realizando análisis ANOVA...")
anova_results, anova_fig = analyze_anova(combined_data)

print('Iniciando KNN...')
knn_model, scaler, best_k, best_accuracy, initial_figs = train_knn_model(combined_data)

print('Plotting accuracy vs k for initial model...')
initial_k_plot, confirmed_initial_k, confirmed_initial_accuracy = plot_model_k_accuracy(combined_data, combined_data.drop('quality', axis=1).columns.tolist())


print('Iniciando análisis de sensibilidad basado en KNN...')
knn_best_model, knn_best_scaler, knn_best_features, knn_best_k, knn_best_accuracy, knn_figures = perform_knn_feature_sensitivity_analysis(combined_data)

print('Plotting accuracy vs k for KNN-based best model...')
knn_best_k_plot, knn_confirmed_best_k, knn_confirmed_best_accuracy = plot_model_k_accuracy(combined_data, knn_best_features)



print('Iniciando análisis de sensibilidad basado en ANOVA...')
anova_best_model, anova_best_scaler, anova_best_features, anova_best_k, anova_best_accuracy, anova_figs = perform_anova_feature_sensitivity_analysis(combined_data, anova_results)

print('Plotting accuracy vs k for ANOVA-based best model...')
anova_best_k_plot, anova_confirmed_best_k, anova_confirmed_best_accuracy = plot_model_k_accuracy(combined_data, anova_best_features)

print('Iniciando KNN simple con dos variables...')
simple_knn_model, simple_scaler, simple_best_k, simple_best_accuracy, simple_conf_matrix = train_simple_knn_model(combined_data)

print('Plotting accuracy vs k for simple model...')
simple_k_plot, simple_confirmed_k, simple_confirmed_accuracy = plot_model_k_accuracy(combined_data, ['volatile acidity', 'total sulfur dioxide'])




















#with +- 1 of prediction
def calculate_relaxed_accuracy(y_true, y_pred, tolerance=1):
    correct = 0
    total = len(y_true)
    
    for true, pred in zip(y_true, y_pred):
        if abs(true - pred) <= tolerance:
            correct += 1
            
    return correct / total

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
    
    colors = ['#c6dbef', '#4292c6']  # Light blue for out-of-tolerance, darker blue for in-tolerance
    cmap = sns.color_palette(colors, as_cmap=True)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=quality_labels,
                yticklabels=quality_labels[::-1],
                mask=~mask)  # Use inverted mask to highlight in-tolerance predictions
    
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

print("\nEvaluating models with ±1 tolerance:")

X = combined_data.drop('quality', axis=1)
y = combined_data['quality']
y_pred = knn_model.predict(scaler.transform(X))
initial_relaxed_acc = calculate_relaxed_accuracy(y, y_pred)
initial_relaxed_cm = plot_relaxed_confusion_matrix(y, y_pred, "Initial KNN Model Confusion Matrix (±1 Tolerance)")
print(f"Initial KNN Model Relaxed Accuracy: {initial_relaxed_acc:.3f}")

X_knn = combined_data[knn_best_features]
y_pred_knn = knn_best_model.predict(knn_best_scaler.transform(X_knn))
knn_relaxed_acc = calculate_relaxed_accuracy(y, y_pred_knn)
knn_relaxed_cm = plot_relaxed_confusion_matrix(y, y_pred_knn, "KNN Feature Selection Best Model Confusion Matrix (±1 Tolerance)")
print(f"KNN Feature Selection Best Model Relaxed Accuracy: {knn_relaxed_acc:.3f}")

X_anova = combined_data[anova_best_features]
y_pred_anova = anova_best_model.predict(anova_best_scaler.transform(X_anova))
anova_relaxed_acc = calculate_relaxed_accuracy(y, y_pred_anova)
anova_relaxed_cm = plot_relaxed_confusion_matrix(y, y_pred_anova, "ANOVA Feature Selection Best Model Confusion Matrix (±1 Tolerance)")
print(f"ANOVA Feature Selection Best Model Relaxed Accuracy: {anova_relaxed_acc:.3f}")

X_simple = combined_data[['volatile acidity', 'total sulfur dioxide']]
y_pred_simple = simple_knn_model.predict(simple_scaler.transform(X_simple))
simple_relaxed_acc = calculate_relaxed_accuracy(y, y_pred_simple)
simple_relaxed_cm = plot_relaxed_confusion_matrix(y, y_pred_simple, "Simple KNN Model Confusion Matrix (±1 Tolerance)")
print(f"Simple KNN Model Relaxed Accuracy: {simple_relaxed_acc:.3f}")

plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')
