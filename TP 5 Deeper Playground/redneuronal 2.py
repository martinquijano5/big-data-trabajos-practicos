import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

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
scaler = StandardScaler()
X1 = scaler.fit_transform(df1[['X1', 'X2']])
X8 = scaler.fit_transform(df8[['X1', 'X2']])

# Obtener las etiquetas
y1 = df1['Type'].values
y8 = df8['Type'].values

# Dividir los datos en conjuntos de entrenamiento y prueba
# Dataset 1
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
# Dataset 8
X8_train, X8_test, y8_train, y8_test = train_test_split(X8, y8, test_size=0.2, random_state=42)

# Imprimir información sobre los conjuntos de datos
print("\nInformación de los conjuntos de datos:")
print("\nDataset 1:")
print(f"Conjunto de entrenamiento: {X1_train.shape[0]} muestras")
print(f"Conjunto de prueba: {X1_test.shape[0]} muestras")
print(f"Proporción de clases en entrenamiento - NARANJA: {np.sum(y1_train == 0)}, AZUL: {np.sum(y1_train == 1)}")
print(f"Proporción de clases en prueba - NARANJA: {np.sum(y1_test == 0)}, AZUL: {np.sum(y1_test == 1)}")

print("\nDataset 8:")
print(f"Conjunto de entrenamiento: {X8_train.shape[0]} muestras")
print(f"Conjunto de prueba: {X8_test.shape[0]} muestras")
print(f"Proporción de clases en entrenamiento - NARANJA: {np.sum(y8_train == 0)}, AZUL: {np.sum(y8_train == 1)}")
print(f"Proporción de clases en prueba - NARANJA: {np.sum(y8_test == 0)}, AZUL: {np.sum(y8_test == 1)}")

# Función para crear el modelo de red neuronal
def create_model():
    model = Sequential([
        # Primera capa oculta con 10 neuronas y activación ReLU
        Dense(10, activation='relu', input_shape=(2,)),
        # Segunda capa oculta con 10 neuronas y activación ReLU
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

# Crear y mostrar el resumen del modelo
model = create_model()
print("\nResumen del modelo de red neuronal:")
model.summary()

# Entrenar el modelo para el dataset 1
print("\nEntrenando modelo para Dataset 1...")
history1 = model.fit(
    X1_train, y1_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluar el modelo en el conjunto de prueba del dataset 1
test_loss1, test_accuracy1 = model.evaluate(X1_test, y1_test)
print(f"\nPrecisión en el conjunto de prueba del Dataset 1: {test_accuracy1:.4f}")

# Crear y entrenar un nuevo modelo para el dataset 8
model8 = create_model()
print("\nEntrenando modelo para Dataset 8...")
history8 = model8.fit(
    X8_train, y8_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluar el modelo en el conjunto de prueba del dataset 8
test_loss8, test_accuracy8 = model8.evaluate(X8_test, y8_test)
print(f"\nPrecisión en el conjunto de prueba del Dataset 8: {test_accuracy8:.4f}")

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

# Visualizar el historial de entrenamiento
plt.figure(figsize=(15, 5))

# Gráfico de pérdida
plt.subplot(1, 2, 1)
plt.plot(history1.history['loss'], label='Entrenamiento Dataset 1')
plt.plot(history1.history['val_loss'], label='Validación Dataset 1')
plt.plot(history8.history['loss'], label='Entrenamiento Dataset 8')
plt.plot(history8.history['val_loss'], label='Validación Dataset 8')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)

# Gráfico de precisión
plt.subplot(1, 2, 2)
plt.plot(history1.history['accuracy'], label='Entrenamiento Dataset 1')
plt.plot(history1.history['val_accuracy'], label='Validación Dataset 1')
plt.plot(history8.history['accuracy'], label='Entrenamiento Dataset 8')
plt.plot(history8.history['val_accuracy'], label='Validación Dataset 8')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Matrices de confusión
plt.figure(figsize=(15, 5))

# Dataset 1
plt.subplot(1, 2, 1)
y1_pred = (model.predict(X1_test) > 0.5).astype(int)
cm1 = confusion_matrix(y1_test, y1_pred)
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Dataset 1')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')

# Dataset 8
plt.subplot(1, 2, 2)
y8_pred = (model8.predict(X8_test) > 0.5).astype(int)
cm8 = confusion_matrix(y8_test, y8_pred)
sns.heatmap(cm8, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Dataset 8')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')

plt.tight_layout()

# Superficies de decisión
plot_decision_boundary(model, X1, y1, 'Superficie de Decisión - Dataset 1')
plot_decision_boundary(model8, X8, y8, 'Superficie de Decisión - Dataset 8')

plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all') 