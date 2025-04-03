import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from itertools import combinations
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


def three_var_model():
    model = Sequential([
        Dense(3, activation='linear', input_shape=(4,), kernel_regularizer=l1(0.03)), #l1 regularizer mete feature importance
        Dense(3, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.01)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Define species colors for consistent visualization
species_colors = {'Iris-setosa': 'blue', 'Iris-versicolor': 'orange', 'Iris-virginica': 'green'}

def train_complete_model(df):
    # Get feature names (excluding 'Id' and target column)
    feature_cols = [col for col in df.columns if col not in ['Id', 'Species']]
    
    # Create label encoder and one-hot encode target
    label_encoder = LabelEncoder()
    encoded_y = label_encoder.fit_transform(df['Species'])
    y_one_hot = to_categorical(encoded_y)
    
    # Extract all features
    X = df[feature_cols].values
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create and train model
    model = three_var_model()
    history = model.fit(
        X_scaled, y_one_hot,
        epochs=300,
        batch_size=16,
        verbose=1,
        validation_split=0.2
    )
    
    # Evaluate on the training data
    loss, accuracy = model.evaluate(X_scaled, y_one_hot, verbose=1)
    
    # Store results
    results = {
        'model': model,
        'history': history,
        'accuracy': accuracy,
        'loss': loss,
        'X_scaled': X_scaled,
        'y_encoded': encoded_y,
        'y_one_hot': y_one_hot,
        'feature_names': feature_cols  # Store feature names for feature importance
    }
    
    return results

def plot_training_results(results):
    # Create separate figure for loss
    plt.figure(figsize=(12, 8))
    plt.title('Pérdida durante el entrenamiento', fontsize=14)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Pérdida', fontsize=12)
    plt.grid(True)
    
    # Plot training loss
    plt.plot(results['history'].history['loss'], color='blue', label='Train loss')
    # Plot validation loss
    plt.plot(results['history'].history['val_loss'], color='blue', alpha=0.5, linestyle='--', label='Val loss')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graficos_dir, 'loss_completo.png'), dpi=300)
    
    # Create separate figure for accuracy
    plt.figure(figsize=(12, 8))
    plt.title('Precisión durante el entrenamiento', fontsize=14)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Precisión', fontsize=12)
    plt.grid(True)
    
    # Plot training accuracy
    plt.plot(results['history'].history['accuracy'], color='green', label='Train accuracy')
    # Plot validation accuracy
    plt.plot(results['history'].history['val_accuracy'], color='green', alpha=0.5, linestyle='--', label='Val accuracy')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graficos_dir, 'accuracy_completo.png'), dpi=300)

def plot_confusion_matrix(results, classes):
    """
    Create and plot confusion matrix for the model
    
    Parameters:
    - results: Dictionary containing the model results
    - classes: List of class names for the confusion matrix
    """
    # Get model and data
    model = results['model']
    X_scaled = results['X_scaled']
    y_encoded = results['y_encoded']
    
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
    plt.title('Matriz de Confusión - Modelo Completo')
    plt.tight_layout()
    plt.savefig(os.path.join(graficos_dir, 'confusion_matrix_completo.png'), dpi=300)

def plot_feature_importance(results):
    """
    Plot feature importance based on L1 regularization weights
    
    Parameters:
    - results: Dictionary containing the model results and feature names
    """
    model = results['model']
    feature_names = results['feature_names']
    
    # Get weights from the first layer (these are influenced by L1 regularization)
    weights = model.layers[0].get_weights()[0]
    
    # Calculate importance as the absolute sum of weights for each feature
    importance = np.sum(np.abs(weights), axis=1)
    
    # Normalize importance
    importance = importance / np.sum(importance)
    
    # Sort features by importance
    indices = np.argsort(importance)
    
    # Create figure for feature importance
    plt.figure(figsize=(10, 6))
    plt.title('Importancia de características (L1 Regularization)', fontsize=14)
    
    # Create the horizontal bar chart
    bars = plt.barh(range(len(indices)), importance[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    
    # Add text labels to the bars
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.01  # Adjust position for better visibility
        plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                 f'{width:.3f}', va='center')
    
    plt.xlabel('Importancia relativa')
    plt.ylabel('Características')
    plt.tight_layout()
    plt.savefig(os.path.join(graficos_dir, 'feature_importance.png'), dpi=300)

def plot_decision_boundary_pca(results, classes):
    """
    Create a decision boundary plot after reducing data to 2D using PCA
    
    Parameters:
    - results: Dictionary containing model results
    - classes: List of class names
    """
    # Get model and data
    model = results['model']
    X_scaled = results['X_scaled']
    y_encoded = results['y_encoded']
    feature_names = results['feature_names']
    
    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    components = pca.components_
    explained_variance = pca.explained_variance_ratio_
    
    # Create a simple table showing the PCA component values
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    
    # Create table data
    table_data = [
        ['', 'PC1', 'PC2', 'Explained Variance'],
    ]
    
    # Add feature data
    for i, feature in enumerate(feature_names):
        row = [feature]
        for j in range(2):  # Just the 2 components
            row.append(f'{components[j, i]:.3f}')
        row.append('')  # Empty cell for alignment
        table_data.append(row)
    
    # Add explained variance in the last row
    table_data.append(['Explained Variance', 
                      f'{explained_variance[0]:.3f}',
                      f'{explained_variance[1]:.3f}',
                      f'{sum(explained_variance):.3f}'])
    
    # Create the table
    table = ax.table(cellText=[r[1:] for r in table_data],
                    rowLabels=[r[0] for r in table_data],
                    colLabels=None,
                    cellLoc='center',
                    loc='center')
    
    # Adjust table formatting
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    plt.title('PCA Component Values')
    plt.tight_layout()
    plt.savefig(os.path.join(graficos_dir, 'pca_table.png'), dpi=300)
    
    # Define the grid for decision boundary
    h = 0.02  # Step size
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Create a mesh grid of points to predict
    grid_points_pca = np.c_[xx.ravel(), yy.ravel()]
    
    # Transform PCA points back to original space using inverse_transform
    grid_points_orig = pca.inverse_transform(grid_points_pca)
    
    # Predict on grid points
    Z = model.predict(grid_points_orig)
    Z = np.argmax(Z, axis=1).reshape(xx.shape)
    
    # Create figure for decision boundary
    plt.figure(figsize=(10, 8))
    
    # Plot decision boundary
    cmap_light = ListedColormap(['#AAAAFF', '#FFAAAA', '#AAFFAA'])
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
    
    # Plot training points
    markers = ['o', 's', '^']
    colors = [species_colors[name] for name in classes]
    
    for i, color in enumerate(colors):
        idx = np.where(y_encoded == i)
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], 
                   c=color, 
                   label=classes[i],
                   marker=markers[i],
                   edgecolor='black',
                   s=80)
    
    plt.title('Frontera de decisión (PCA 2D)')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    
    # Add explained variance ratio to the plot
    plt.text(0.02, 0.98, f'PC1 var: {explained_variance[0]:.2f}\nPC2 var: {explained_variance[1]:.2f}',
             transform=plt.gca().transAxes, va='top')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graficos_dir, 'decision_boundary_pca.png'), dpi=300)

# Set up directories
current_dir = os.path.dirname(os.path.abspath(__file__))
# Create graficos directory if it doesn't exist
graficos_dir = os.path.join(current_dir, 'graficos', 'modelo_completo')
os.makedirs(graficos_dir, exist_ok=True)

# Load data
iris_df = pd.read_csv(os.path.join(current_dir, 'iris.csv'))

# Get class names for confusion matrix
class_names = iris_df['Species'].unique()

# Create and train the model with all features
results = train_complete_model(iris_df)

# Print results
print(f"Model Accuracy: {results['accuracy']:.4f}")

# Plot training results
plot_training_results(results)

# Plot confusion matrix
plot_confusion_matrix(results, class_names)

# Plot feature importance
plot_feature_importance(results)

# Plot decision boundary with PCA
plot_decision_boundary_pca(results, class_names)

plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')


