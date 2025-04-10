import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from itertools import combinations
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap


# Define species colors for consistent visualization
species_colors = {'Iris-setosa': 'blue', 'Iris-versicolor': 'orange', 'Iris-virginica': 'green'}

def simple_model():
    model = Sequential([
        Dense(3, activation='linear', input_shape=(2,)), #uso linear porque la separacion entre las clases es lineal, relu sirve para relaciones no lineales
        Dense(3, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.01)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def create_simple_models(df):
    # Get feature names (excluding 'Id' and target column)
    feature_cols = [col for col in df.columns if col not in ['Id', 'Species']]
    
    # Create label encoder and one-hot encode target
    label_encoder = LabelEncoder()
    encoded_y = label_encoder.fit_transform(df['Species'])
    y_one_hot = to_categorical(encoded_y)
    
    # Dictionary to store results
    results = {}
    
    # Generate all possible 2-variable combinations
    feature_combinations = list(combinations(feature_cols, 2))
    print(f"Combinaciones de features: {feature_combinations}")
    
    # For each combination, train a model
    for combinacion in feature_combinations:
        print(f"Training model with features: {combinacion}")
        
        # Extract only these two features
        X_subset = df[list(combinacion)].values
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_subset)
        
        # Create and train model
        model = simple_model()
        history = model.fit(
            X_scaled, y_one_hot,
            epochs=100,
            batch_size=16,
            verbose=1,
            validation_split=0.2
        )
        
        # Evaluate on the training data
        loss, accuracy = model.evaluate(X_scaled, y_one_hot, verbose=1)
        
        # Store results
        results[combinacion] = {
            'model': model,
            'history': history,
            'accuracy': accuracy,
            'loss': loss,
            'X_scaled': X_scaled,
            'y_encoded': encoded_y,
            'y_one_hot': y_one_hot
        }
    
    return results

def plot_training_results(results):
    # Define colors for each combination
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    
    # Create separate figure for loss
    plt.figure(figsize=(12, 8))
    plt.title('Pérdida durante el entrenamiento (K-Fold)', fontsize=14)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Pérdida', fontsize=12)
    plt.grid(True)
    
    # Plot each combination for loss
    for i, (combinacion, result) in enumerate(results.items()):
        history = result['history']
        combinacion_name = f"{combinacion[0]}-{combinacion[1]}"
        
        # Plot training loss
        plt.plot(history.history['loss'], color=colors[i], label=f'Train {combinacion_name}')
        # Plot validation loss
        plt.plot(history.history['val_loss'], color=colors[i], alpha=0.5, linestyle='--', label=f'Val {combinacion_name}')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graficos_dir, 'loss.png'), dpi=300)
    
    # Create separate figure for accuracy
    plt.figure(figsize=(12, 8))
    plt.title('Precisión durante el entrenamiento (K-Fold)', fontsize=14)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Precisión', fontsize=12)
    plt.grid(True)
    
    # Plot each combination for accuracy
    for i, (combinacion, result) in enumerate(results.items()):
        history = result['history']
        combinacion_name = f"{combinacion[0]}-{combinacion[1]}"
        
        # Plot training accuracy
        plt.plot(history.history['accuracy'], color=colors[i], label=f'Train {combinacion_name}')
        # Plot validation accuracy
        plt.plot(history.history['val_accuracy'], color=colors[i], alpha=0.5, linestyle='--', label=f'Val {combinacion_name}')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graficos_dir, 'accuracy.png'), dpi=300)

def plot_confusion_matrices(results, classes):
    """
    Create and plot confusion matrices for each model
    
    Parameters:
    - results: Dictionary containing the model results
    - classes: List of class names for the confusion matrix
    """
    for combinacion, result in results.items():
        # Get model and data
        model = result['model']
        X_scaled = result['X_scaled']
        y_encoded = result['y_encoded']
        
        # Get predictions
        y_pred_proba = model.predict(X_scaled)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_encoded, y_pred)
        
        # Create figure and plot
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        
        # Set title and save
        combinacion_name = f"{combinacion[0]}-{combinacion[1]}"
        plt.title(f'Matriz de Confusión - Modelo {combinacion_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(graficos_dir, f'confusion_matrix_{combinacion_name}.png'), dpi=300)

def plot_decision_boundaries(results, class_names):
    """
    Plot decision boundaries for each model
    
    Parameters:
    - results: Dictionary containing the model results
    - class_names: List of class names
    """
    # Define a meshgrid to visualize the decision boundaries
    def make_meshgrid(x, y, h=.02):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy
    
    # Use the predefined species colors
    colors = [species_colors[name] for name in class_names]
    
    # Create a custom colormap that matches the species colors
    custom_cmap = ListedColormap([species_colors[name] for name in class_names])
    
    # Loop through each model/feature combination
    for combinacion, result in results.items():
        # Get model and data
        model = result['model']
        X_scaled = result['X_scaled']
        y_encoded = result['y_encoded']
        
        # Extract the two features
        feature_one, feature_two = combinacion
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create meshgrid
        X0, X1 = X_scaled[:, 0], X_scaled[:, 1]
        xx, yy = make_meshgrid(X0, X1)
        
        # Reshape meshgrid for prediction
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Predict class for each point in the meshgrid
        Z = model.predict(grid_points)
        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary using the custom colormap
        plt.contourf(xx, yy, Z, cmap=custom_cmap, alpha=0.3)
        
        # Plot the training points
        for i, color in zip(range(len(class_names)), colors):
            idx = np.where(y_encoded == i)
            plt.scatter(X_scaled[idx, 0], X_scaled[idx, 1], c=color, 
                        label=class_names[i], edgecolor='black', s=40)
        
        # Set plot title and labels
        combinacion_name = f"{combinacion[0]}-{combinacion[1]}"
        plt.title(f'Decision Boundary - Model {combinacion_name}')
        plt.xlabel(feature_one)
        plt.ylabel(feature_two)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(graficos_dir, f'decision_boundary_{combinacion_name}.png'), dpi=300)

current_dir = os.path.dirname(os.path.abspath(__file__))
# Create graficos directory if it doesn't exist
graficos_dir = os.path.join(current_dir, 'graficos', 'modelos_simples')
os.makedirs(graficos_dir, exist_ok=True)

iris_df = pd.read_csv(os.path.join(current_dir, 'iris.csv'))

X = iris_df.drop(['Species','Id'], axis=1)
y = iris_df['Species']

# Get class names for confusion matrix
class_names = iris_df['Species'].unique()

# Create and train models for each feature combination
results = create_simple_models(iris_df)

# Print results
for combinacion, result in results.items():
    print(f"Features: {combinacion}, Accuracy: {result['accuracy']:.4f}")

# Plot training results
plot_training_results(results)

# Plot confusion matrices
plot_confusion_matrices(results, class_names)

# Plot decision boundaries
plot_decision_boundaries(results, class_names)

plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')