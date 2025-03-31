import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from tensorflow.keras.regularizers import l2

# Función para crear el modelo de red neuronal de dos capas ocultas
def create_model_8():
    model = Sequential([
        # Primera capa oculta con 20 neuronas y activación ReLU
        Dense(10, activation='relu', input_shape=(2,)),
        # Segunda capa oculta con 15 neuronas y activación ReLU
        Dense(10, activation='relu'),
        # Tercera capa oculta con 10 neuronas y activación ReLU
        Dense(10, activation='relu'),


        # Capa de salida con 1 neurona y activación sigmoid
        Dense(1, activation='sigmoid')
    ])
    
    # Compilar el modelo
    model.compile(
        optimizer=Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# Función para crear el modelo de red neuronal de tres capas ocultas
def create_model_1():
    model = Sequential([
        # Primera capa oculta con 10 neuronas y activación ReLU
        Dense(50, activation='relu', input_shape=(2,)),
        # Segunda capa oculta con 10 neuronas y activación ReLU
        Dense(10, activation='relu'),
        # Tercera capa oculta con 10 neuronas y activación ReLU
        Dense(10, activation='relu'),
        # Cuarta capa oculta con 10 neuronas y activación ReLU
        Dense(10, activation='relu'),
        # Quinta capa oculta con 10 neuronas y activación ReLU
        Dense(10, activation='relu'),
        # Sexta capa oculta con 10 neuronas y activación ReLU
        Dense(10, activation='relu'),
        # Septima capa oculta con 10 neuronas y activación ReLU
        Dense(10, activation='relu'),

        

        # Capa de salida con 1 neurona y activación sigmoid
        Dense(1, activation='sigmoid')
    ])
    
    # Compilar el modelo
    model.compile(
        optimizer=Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# Obtener la ruta del directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path_1 = os.path.join(current_dir, 'dataset_1.csv')
csv_path_8 = os.path.join(current_dir, 'dataset_8.csv')

# Leer los archivos CSV con el formato decimal correcto
df1 = pd.read_csv(csv_path_1, decimal=',')
df8 = pd.read_csv(csv_path_8, decimal=',')

# Verificar y eliminar datos faltantes si existen
df1 = df1.dropna()
df8 = df8.dropna()

# Convertir etiquetas a números (NARANJA = 0, AZUL = 1)
df1['Type'] = df1['Type'].map({'NARANJA': 0, 'AZUL': 1})
df8['Type'] = df8['Type'].map({'NARANJA': 0, 'AZUL': 1})

# Normalizar las características usando StandardScaler
scaler1 = StandardScaler()
scaler8 = StandardScaler()
X1 = scaler1.fit_transform(df1[['X1', 'X2']])
X8 = scaler8.fit_transform(df8[['X1', 'X2']])

# Obtener las etiquetas
y1 = df1['Type'].values
y8 = df8['Type'].values

# Configuración de K-Fold Cross Validation
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Imprimir información sobre los conjuntos de datos
print("\nInformación de los conjuntos de datos:")
print("\nDataset 1:")
print(f"Total de muestras: {X1.shape[0]}")
print(f"Proporción de clases - NARANJA: {np.sum(y1 == 0)}, AZUL: {np.sum(y1 == 1)}")

print("\nDataset 8:")
print(f"Total de muestras: {X8.shape[0]}")
print(f"Proporción de clases - NARANJA: {np.sum(y8 == 0)}, AZUL: {np.sum(y8 == 1)}")
print(f"\nNúmero de folds: {k_folds}")


# Crear y mostrar el resumen del modelo
print("\nResumen del modelo de red neuronal:")
create_model_8().summary()

# Listas para almacenar resultados del Dataset 1
fold_accuracy_1 = []
fold_loss_1 = []
all_histories_1 = []
y_true_all_1 = []
y_pred_all_1 = []

# Realizar k-fold cross validation para el Dataset 1
print("\nRealizando K-Fold Cross Validation para Dataset 1...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X1)):
    print(f"\nEntrenando fold {fold+1}/{k_folds}")
    
    # Dividir datos en conjuntos de entrenamiento y validación
    X_train, X_val = X1[train_idx], X1[val_idx]
    y_train, y_val = y1[train_idx], y1[val_idx]
    
    # Crear y entrenar el modelo
    model = create_model_1()
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )
    all_histories_1.append(history)
    
    # Evaluar el modelo
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    fold_loss_1.append(val_loss)
    fold_accuracy_1.append(val_acc)
    
    # Guardar predicciones para la matriz de confusión
    y_pred = (model.predict(X_val) > 0.5).astype(int)
    y_true_all_1.extend(y_val)
    y_pred_all_1.extend(y_pred)
    
    print(f"Fold {fold+1} - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

# Mostrar resultados de la validación cruzada para Dataset 1
print("\nResultados de Cross Validation para Dataset 1:")
print(f"Precisión promedio: {np.mean(fold_accuracy_1):.4f} ± {np.std(fold_accuracy_1):.4f}")
print(f"Loss promedio: {np.mean(fold_loss_1):.4f} ± {np.std(fold_loss_1):.4f}")

# Listas para almacenar resultados del Dataset 8
fold_accuracy_8 = []
fold_loss_8 = []
all_histories_8 = []
y_true_all_8 = []
y_pred_all_8 = []

# Realizar k-fold cross validation para el Dataset 8
print("\nRealizando K-Fold Cross Validation para Dataset 8...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X8)):
    print(f"\nEntrenando fold {fold+1}/{k_folds}")
    
    # Dividir datos en conjuntos de entrenamiento y validación
    X_train, X_val = X8[train_idx], X8[val_idx]
    y_train, y_val = y8[train_idx], y8[val_idx]
    
    # Crear y entrenar el modelo
    model = create_model_8()
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )
    all_histories_8.append(history)
    
    # Evaluar el modelo
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    fold_loss_8.append(val_loss)
    fold_accuracy_8.append(val_acc)
    
    # Guardar predicciones para la matriz de confusión
    y_pred = (model.predict(X_val) > 0.5).astype(int)
    y_true_all_8.extend(y_val)
    y_pred_all_8.extend(y_pred)
    
    print(f"Fold {fold+1} - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

# Mostrar resultados de la validación cruzada para Dataset 8
print("\nResultados de Cross Validation para Dataset 8:")
print(f"Precisión promedio: {np.mean(fold_accuracy_8):.4f} ± {np.std(fold_accuracy_8):.4f}")
print(f"Loss promedio: {np.mean(fold_loss_8):.4f} ± {np.std(fold_loss_8):.4f}")

# Comparación directa entre ambos datasets
print("\nComparación de resultados:")
print(f"Dataset 1 - Precisión: {np.mean(fold_accuracy_1):.4f} ± {np.std(fold_accuracy_1):.4f}")
print(f"Dataset 8 - Precisión: {np.mean(fold_accuracy_8):.4f} ± {np.std(fold_accuracy_8):.4f}")

# Construir modelos finales con todos los datos
print("\nEntrenando modelos finales con todos los datos...")

# Modelo final para Dataset 1
final_model_1 = create_model_1()
final_history_1 = final_model_1.fit(
    X1, y1,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Modelo final para Dataset 8
final_model_8 = create_model_8()
final_history_8 = final_model_8.fit(
    X8, y8,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Función para crear la superficie de decisión
def plot_decision_boundary(model, X, y, title):
    # Crear una malla de puntos
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Predecir para cada punto de la malla
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Crear el gráfico
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', alpha=0.8)
    plt.colorbar()
    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')

# Visualizar el historial de entrenamiento de los folds
plt.figure(figsize=(15, 7))

# Gráfico de pérdida promedio por época
plt.subplot(1, 2, 1)

# Dataset 1 - Líneas de promedio
mean_train_loss_1 = np.mean([h.history['loss'] for h in all_histories_1], axis=0)
mean_val_loss_1 = np.mean([h.history['val_loss'] for h in all_histories_1], axis=0)
plt.plot(mean_train_loss_1, label='Train D1', color='blue', linewidth=2)
plt.plot(mean_val_loss_1, label='Val D1', color='lightblue', linewidth=2)

# Dataset 8 - Líneas de promedio
mean_train_loss_8 = np.mean([h.history['loss'] for h in all_histories_8], axis=0)
mean_val_loss_8 = np.mean([h.history['val_loss'] for h in all_histories_8], axis=0)
plt.plot(mean_train_loss_8, label='Train D8', color='red', linewidth=2)
plt.plot(mean_val_loss_8, label='Val D8', color='lightcoral', linewidth=2)

plt.title('Pérdida durante el entrenamiento (K-Fold)')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)

# Gráfico de precisión promedio por época
plt.subplot(1, 2, 2)

# Dataset 1 - Líneas de promedio
mean_train_acc_1 = np.mean([h.history['accuracy'] for h in all_histories_1], axis=0)
mean_val_acc_1 = np.mean([h.history['val_accuracy'] for h in all_histories_1], axis=0)
plt.plot(mean_train_acc_1, label='Train D1', color='blue', linewidth=2)
plt.plot(mean_val_acc_1, label='Val D1', color='lightblue', linewidth=2)

# Dataset 8 - Líneas de promedio
mean_train_acc_8 = np.mean([h.history['accuracy'] for h in all_histories_8], axis=0)
mean_val_acc_8 = np.mean([h.history['val_accuracy'] for h in all_histories_8], axis=0)
plt.plot(mean_train_acc_8, label='Train D8', color='red', linewidth=2)
plt.plot(mean_val_acc_8, label='Val D8', color='lightcoral', linewidth=2)

plt.title('Precisión durante el entrenamiento (K-Fold)')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Matrices de confusión para los modelos finales
plt.figure(figsize=(15, 6))

# Dataset 1 - usando el modelo final
y_pred_final_1 = (final_model_1.predict(X1) > 0.5).astype(int)
cm1_final = confusion_matrix(y1, y_pred_final_1)
plt.subplot(1, 2, 1)
sns.heatmap(cm1_final, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión Final - Dataset 1')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')

# Dataset 8 - usando el modelo final
y_pred_final_8 = (final_model_8.predict(X8) > 0.5).astype(int)
cm8_final = confusion_matrix(y8, y_pred_final_8)
plt.subplot(1, 2, 2)
sns.heatmap(cm8_final, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión Final - Dataset 8')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')

plt.tight_layout()

# Superficies de decisión con los modelos finales
plot_decision_boundary(final_model_1, X1, y1, 'Superficie de Decisión - Dataset 1')
plot_decision_boundary(final_model_8, X8, y8, 'Superficie de Decisión - Dataset 8')

plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all') 