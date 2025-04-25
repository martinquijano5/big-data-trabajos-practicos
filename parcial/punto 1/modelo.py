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
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy import stats


color_map = {-1: '#66b3ff', 0: '#ff9999', 1: '#99ff99'}

def one_hot_encode_custom(y):
    """
    Custom one-hot encoding for labels -1, 0, and 1
    Returns one-hot encoded array preserving the original label order
    """
    # Create a mapping of original values to indices
    value_to_index = {-1: 0, 0: 1, 1: 2}
    
    # Initialize zero matrix with shape (n_samples, 3)
    n_samples = len(y)
    one_hot = np.zeros((n_samples, 3))
    
    # Fill the one-hot matrix
    for i, val in enumerate(y):
        one_hot[i, value_to_index[val]] = 1
    
    return one_hot

def full_model():
    model = Sequential([
        Dense(8, activation='relu', input_shape=(2,)),
        Dense(6, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.01)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_complete_model(df):
    X = df[['x', 'y']].values
    y = df['z'].values
    
    # Apply one-hot encoding to the target variable
    y_encoded = one_hot_encode_custom(y)
    
    # Create and train model
    model = full_model()
    history = model.fit(
        X, y_encoded,
        epochs=300,
        batch_size=32,
        verbose=1,
        validation_split=0.2
    )
    
    # Get predictions
    y_pred_proba = model.predict(X)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_encoded, axis=1)
    
    # Convert predictions back to original labels (-1, 0, 1)
    index_to_value = {0: -1, 1: 0, 2: 1}
    y_pred_original = np.array([index_to_value[i] for i in y_pred])
    
    # Calculate metrics
    precision = precision_score(y, y_pred_original, average='weighted')
    recall = recall_score(y, y_pred_original, average='weighted')
    f1 = f1_score(y, y_pred_original, average='weighted')
    
    # Calculate AUC (one-vs-rest approach for multiclass)
    auc_scores = []
    for i in range(3):
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = y_pred_proba[:, i]
        auc_scores.append(roc_auc_score(y_true_binary, y_pred_binary))
    auc = np.mean(auc_scores)
    
    # Calculate KS statistic
    ks_stats = []
    for i in range(3):
        pos_scores = y_pred_proba[y_true == i, i]
        neg_scores = y_pred_proba[y_true != i, i]
        ks_stat = stats.ks_2samp(pos_scores, neg_scores)[0]
        ks_stats.append(ks_stat)
    ks = np.mean(ks_stats)
    
    # Evaluate on the training data for loss and accuracy
    loss, accuracy = model.evaluate(X, y_encoded, verbose=0)
    
    # Print all metrics
    print("\nModel Performance Metrics:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC (avg): {auc:.4f}")
    print(f"KS Statistic (avg): {ks:.4f}")
    
    # Store results
    results = {
        'model': model,
        'history': history,
        'accuracy': accuracy,
        'loss': loss,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'ks': ks,
        'X': X,
        'y': y_encoded,
        'feature_names': ['x', 'y']
    }
    
    return results

def plot_results_table(results):
    # Define the metrics to display
    metrics = {
        'Loss': results['loss'],
        'Accuracy': results['accuracy'],
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1 Score': results['f1_score'],
        'AUC': results['auc'],
        'KS Statistic': results['ks']
    }
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table data
    table_data = [[metric, f"{value:.4f}"] for metric, value in metrics.items()]
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Metric', 'Value'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.5, 0.3])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style header
    for j in range(2):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', weight='bold')
    
    # Add alternating row colors
    for i in range(len(metrics)):
        if i % 2:
            for j in range(2):
                table[(i + 1, j)].set_facecolor('#E9EDF4')
    
    plt.title('Model Performance Metrics', pad=20)
    
    # Save the figure
    plt.savefig(os.path.join(graficos_dir, 'results_table.png'), 
                bbox_inches='tight', 
                dpi=300)
    
    return fig, ax

def plot_loss_epochs(results):
    # Get training history
    history = results['history']
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot training & validation loss
    plt.plot(history.history['loss'], label='Train loss', color='blue')
    plt.plot(history.history['val_loss'], label='Val loss', color='blue', linestyle='--', alpha=0.7)
    
    # Customize the plot
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the figure
    plt.savefig(os.path.join(graficos_dir, 'loss_epochs.png'), 
                bbox_inches='tight', 
                dpi=300)
    
    return plt.gcf(), plt.gca()

def plot_accuracy_epochs(results):
    # Get training history
    history = results['history']
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot training & validation accuracy
    plt.plot(history.history['accuracy'], label='Train accuracy', color='green')
    plt.plot(history.history['val_accuracy'], label='Val accuracy', color='green', linestyle='--', alpha=0.7)
    
    # Customize the plot
    plt.title('Precisión durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Set y-axis limits to match the example (0 to 1)
    plt.ylim(0, 1)
    
    # Save the figure
    plt.savefig(os.path.join(graficos_dir, 'accuracy_epochs.png'), 
                bbox_inches='tight', 
                dpi=300)
    
    return plt.gcf(), plt.gca()

def plot_confusion_matrix(results):
    # Get model predictions
    model = results['model']
    X = results['X']
    y_encoded = results['y']
    
    # Get predictions
    y_pred_proba = model.predict(X)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_encoded, axis=1)
    
    # Convert back to original labels (-1, 0, 1)
    index_to_value = {0: -1, 1: 0, 2: 1}
    y_pred_original = np.array([index_to_value[i] for i in y_pred])
    y_true_original = np.array([index_to_value[i] for i in y_true])
    
    # Create confusion matrix
    cm = confusion_matrix(y_true_original, y_pred_original)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Create confusion matrix display
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[-1, 0, 1]
    )
    
    # Plot confusion matrix
    disp.plot(cmap='Blues', values_format='d')
    
    # Customize the plot
    plt.title('Matriz de Confusión')
    
    # Save the figure
    plt.savefig(os.path.join(graficos_dir, 'confusion_matrix.png'), 
                bbox_inches='tight', 
                dpi=300)
    
    return plt.gcf(), plt.gca()

def plot_feature_importance(results):
    # Get model and feature names
    model = results['model']
    feature_names = results['feature_names']
    
    # Get weights from the first layer (where L1 regularization was applied)
    weights = np.abs(model.layers[0].get_weights()[0])
    
    # Calculate feature importance as the sum of absolute weights for each feature
    feature_importance = np.sum(weights, axis=1)
    
    # Normalize feature importance
    feature_importance = feature_importance / np.sum(feature_importance)
    
    # Create data for the table
    importance_data = list(zip(feature_names, feature_importance))
    importance_data.sort(key=lambda x: x[1], reverse=True)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    table_data = [[name, f"{importance:.4f}"] for name, importance in importance_data]
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Feature', 'Importance'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.5, 0.3])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style header
    for j in range(2):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', weight='bold')
    
    # Add alternating row colors
    for i in range(len(feature_names)):
        if i % 2:
            for j in range(2):
                table[(i + 1, j)].set_facecolor('#E9EDF4')
    
    plt.title('Importancia de Features', pad=20)
    
    # Save the figure
    plt.savefig(os.path.join(graficos_dir, 'feature_importance.png'), 
                bbox_inches='tight', 
                dpi=300)
    
    return fig, ax

def plot_decision_boundary(results):
    # Get data and model
    X = results['X']
    model = results['model']
    
    # Create a mesh grid with higher resolution
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                        np.linspace(y_min, y_max, 300))
    
    # Get predictions for all mesh points
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)  # Reshape to match the grid
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create custom colormap for the decision regions
    colors = [color_map[-1], color_map[0], color_map[1]]
    custom_cmap = ListedColormap(colors)
    
    # Plot decision boundary with custom colormap
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=custom_cmap)
    
    # Plot training points
    for value in [-1, 0, 1]:
        mask = df['z'] == value
        plt.scatter(df[mask]['x'], df[mask]['y'], 
                   c=[color_map[value]], 
                   label=f'Clase {value}',
                   alpha=0.6,
                   edgecolors='k',
                   s=50)
    
    # Customize the plot
    plt.title('Frontera de decisión')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save the figure
    plt.savefig(os.path.join(graficos_dir, 'decision_boundary.png'), 
                bbox_inches='tight', 
                dpi=300)
    
    return plt.gcf(), plt.gca()

# Set up directories
current_dir = os.path.dirname(os.path.abspath(__file__))
# Create graficos directory if it doesn't exist
graficos_dir = os.path.join(current_dir, 'graficos', 'modelo')
os.makedirs(graficos_dir, exist_ok=True)

# Load data
df = pd.read_excel('parcial/punto 1/Ej_1_A337_2025.xlsx')

# Get class names for confusion matrix
class_names = df['z'].unique()

# Create and train the model with all features
results = train_complete_model(df)


plot_results_table(results)
plot_loss_epochs(results)
plot_accuracy_epochs(results)
plot_confusion_matrix(results)
plot_feature_importance(results)
plot_decision_boundary(results)





plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')