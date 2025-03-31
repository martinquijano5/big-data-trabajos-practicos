import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

# Obtener la ruta del directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path_1 = os.path.join(current_dir, 'dataset_1.csv')
csv_path_8 = os.path.join(current_dir, 'dataset_8.csv')

# Leer los archivos CSV con el formato decimal correcto
df1 = pd.read_csv(csv_path_1, decimal=',')
df8 = pd.read_csv(csv_path_8, decimal=',')

# Verificar y eliminar datos faltantes si existen
print("\nDatos faltantes en Dataset 1:")
print(df1.isnull().sum())
print("\nDatos faltantes en Dataset 8:")
print(df8.isnull().sum())

df1 = df1.dropna()
df8 = df8.dropna()

# Convertir etiquetas a números (NARANJA = 0, AZUL = 1)
df1['Type'] = df1['Type'].map({'NARANJA': 0, 'AZUL': 1})
df8['Type'] = df8['Type'].map({'NARANJA': 0, 'AZUL': 1})

# Normalizar las características usando StandardScaler, aunque por como es este dataset no es necesario
scaler = StandardScaler()
X1 = scaler.fit_transform(df1[['X1', 'X2']])
X8 = scaler.fit_transform(df8[['X1', 'X2']])

# Obtener las etiquetas
y1 = df1['Type'].values
y8 = df8['Type'].values

# Verificar la forma de los datos
print("\nForma de los datos:")
print(f"Dataset 1 - X: {X1.shape}, y: {y1.shape}")
print(f"Dataset 8 - X: {X8.shape}, y: {y8.shape}")

# Verificar la normalización
print("\nVerificación de la normalización:")
print("\nDataset 1:")
print(f"Media X1: {X1[:, 0].mean():.6f}, Desv. Est. X1: {X1[:, 0].std():.6f}")
print(f"Media X2: {X1[:, 1].mean():.6f}, Desv. Est. X2: {X1[:, 1].std():.6f}")

print("\nDataset 8:")
print(f"Media X1: {X8[:, 0].mean():.6f}, Desv. Est. X1: {X8[:, 0].std():.6f}")
print(f"Media X2: {X8[:, 1].mean():.6f}, Desv. Est. X2: {X8[:, 1].std():.6f}")

# Verificar la codificación de clases
print("\nDistribución de clases:")
print("\nDataset 1:")
print("NARANJA (0):", np.sum(y1 == 0))
print("AZUL (1):", np.sum(y1 == 1))

print("\nDataset 8:")
print("NARANJA (0):", np.sum(y8 == 0))
print("AZUL (1):", np.sum(y8 == 1))

# Figura 1: Gráficos de torta
plt.figure(figsize=(15, 6))

# Gráfico de torta dataset 1
plt.subplot(1, 2, 1)
valores1 = [np.sum(y1 == 0), np.sum(y1 == 1)]
plt.pie(valores1, 
        labels=['NARANJA', 'AZUL'],
        colors=['orange', 'blue'],
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.05, 0.05))
plt.title('Distribución de Clases - Dataset 1', fontsize=14, pad=20, fontweight='bold')

# Gráfico de torta dataset 8
plt.subplot(1, 2, 2)
valores8 = [np.sum(y8 == 0), np.sum(y8 == 1)]
plt.pie(valores8, 
        labels=['NARANJA', 'AZUL'],
        colors=['orange', 'blue'],
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.05, 0.05))
plt.title('Distribución de Clases - Dataset 8', fontsize=14, pad=20, fontweight='bold')
plt.tight_layout()

# Figura 2: Gráficos de dispersión
plt.figure(figsize=(15, 6))

# Dataset 1
plt.subplot(1, 2, 1)
colors1 = ['orange' if label == 0 else 'blue' for label in y1]
scatter1 = plt.scatter(X1[:, 0], X1[:, 1], 
                     c=colors1,
                     alpha=0.7,
                     s=70,
                     edgecolor='black',
                     linewidth=0.5)

plt.xlabel('X', fontsize=12, fontweight='bold')
plt.ylabel('Y', fontsize=12, fontweight='bold')
plt.title('Gráfico de Dispersión - Dataset 1', fontsize=14, pad=20, fontweight='bold')
plt.grid(True, alpha=0.2, linestyle='--')

# Dataset 8
plt.subplot(1, 2, 2)
colors8 = ['orange' if label == 0 else 'blue' for label in y8]
scatter8 = plt.scatter(X8[:, 0], X8[:, 1], 
                     c=colors8,
                     alpha=0.7,
                     s=70,
                     edgecolor='black',
                     linewidth=0.5)

plt.xlabel('X', fontsize=12, fontweight='bold')
plt.ylabel('Y', fontsize=12, fontweight='bold')
plt.title('Gráfico de Dispersión - Dataset 8', fontsize=14, pad=20, fontweight='bold')
plt.grid(True, alpha=0.2, linestyle='--')

plt.tight_layout()


plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')