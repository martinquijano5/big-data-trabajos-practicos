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
from mpl_toolkits.mplot3d import Axes3D


# Define species colors for consistent visualization
species_colors = {'Iris-setosa': 'blue', 'Iris-versicolor': 'orange', 'Iris-virginica': 'green'}

def three_var_model():
    model = Sequential([
        Dense(3, activation='linear', input_shape=(3,)),
        Dense(3, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.01)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def create_three_var_models(df):
    # Get feature names (excluding 'Id' and target column)
    feature_cols = [col for col in df.columns if col not in ['Id', 'Species']]
    
    # Create label encoder and one-hot encode target
    label_encoder = LabelEncoder()
    encoded_y = label_encoder.fit_transform(df['Species'])
    y_one_hot = to_categorical(encoded_y)
    
    # Dictionary to store results
    results = {}
    
    # Generate all possible 3-variable combinations instead of 2
    feature_combinations = list(combinations(feature_cols, 3))
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
    for i, (combo, result) in enumerate(results.items()):
        history = result['history']
        combo_name = f"{combo[0]}-{combo[1]}-{combo[2]}" # Updated to include third feature
        
        # Plot training loss
        plt.plot(history.history['loss'], color=colors[i], label=f'Train {combo_name}')
        # Plot validation loss
        plt.plot(history.history['val_loss'], color=colors[i], alpha=0.5, linestyle='--', label=f'Val {combo_name}')
    
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
    for i, (combo, result) in enumerate(results.items()):
        history = result['history']
        combo_name = f"{combo[0]}-{combo[1]}-{combo[2]}" # Updated to include third feature
        
        # Plot training accuracy
        plt.plot(history.history['accuracy'], color=colors[i], label=f'Train {combo_name}')
        # Plot validation accuracy
        plt.plot(history.history['val_accuracy'], color=colors[i], alpha=0.5, linestyle='--', label=f'Val {combo_name}')
    
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
    for combo, result in results.items():
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
        combo_name = f"{combo[0]}-{combo[1]}-{combo[2]}" # Updated to include third feature
        plt.title(f'Matriz de Confusión - Modelo {combo_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(graficos_dir, f'confusion_matrix_{combo_name}.png'), dpi=300)

def plot_decision_boundaries(results, class_names):
    """
    Plot 3D decision boundaries for each model
    
    Parameters:
    - results: Dictionary containing the model results
    - class_names: List of class names
    """
    
    # Use the predefined species colors
    colors = [species_colors[name] for name in class_names]
    
    # Loop through each model/feature combination
    for combo, result in results.items():
        # Get model and data
        model = result['model']
        X_scaled = result['X_scaled']
        y_encoded = result['y_encoded']
        
        # Extract the three features
        feature_one, feature_two, feature_three = combo
        
        # Create a 3D figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Define the grid for 3D space with fewer points for better performance
        x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
        y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
        z_min, z_max = X_scaled[:, 2].min() - 0.5, X_scaled[:, 2].max() + 0.5
        
        # Create scatter plots for the data points by class
        for i, color in zip(range(len(class_names)), colors):
            idx = np.where(y_encoded == i)
            ax.scatter(
                X_scaled[idx, 0], X_scaled[idx, 1], X_scaled[idx, 2],
                c=color, label=class_names[i], edgecolor='black', s=70, alpha=1
            )
        
        # Create several slices to visualize the decision boundaries
        # We'll use a coarser grid for better performance
        h = 0.25
        
        # Create z-slices (XY planes at different z values)
        for z_val in np.linspace(z_min, z_max, 4):
            xx_2d, yy_2d = np.meshgrid(
                np.arange(x_min, x_max, h),
                np.arange(y_min, y_max, h)
            )
            
            # Points for this z-slice
            grid_points = np.zeros((xx_2d.size, 3))
            grid_points[:, 0] = xx_2d.ravel()
            grid_points[:, 1] = yy_2d.ravel()
            grid_points[:, 2] = z_val
            
            # Get predictions
            Z = model.predict(grid_points)
            Z = np.argmax(Z, axis=1).reshape(xx_2d.shape)
            
            # Plot each class region as a separate surface with its color
            for i, color in zip(range(len(class_names)), colors):
                mask = Z == i
                if np.any(mask):
                    x_plot = xx_2d[mask]
                    y_plot = yy_2d[mask]
                    z_plot = np.full_like(x_plot, z_val)
                    
                    ax.scatter(x_plot, y_plot, z_plot, c=color, alpha=0.1, s=10)
        
        # Create x-slices (YZ planes at different x values)
        for x_val in np.linspace(x_min, x_max, 4):
            yy_2d, zz_2d = np.meshgrid(
                np.arange(y_min, y_max, h),
                np.arange(z_min, z_max, h)
            )
            
            # Points for this x-slice
            grid_points = np.zeros((yy_2d.size, 3))
            grid_points[:, 0] = x_val
            grid_points[:, 1] = yy_2d.ravel()
            grid_points[:, 2] = zz_2d.ravel()
            
            # Get predictions
            Z = model.predict(grid_points)
            Z = np.argmax(Z, axis=1).reshape(yy_2d.shape)
            
            # Plot each class region
            for i, color in zip(range(len(class_names)), colors):
                mask = Z == i
                if np.any(mask):
                    y_plot = yy_2d[mask]
                    z_plot = zz_2d[mask]
                    x_plot = np.full_like(y_plot, x_val)
                    
                    ax.scatter(x_plot, y_plot, z_plot, c=color, alpha=0.1, s=10)
        
        # Create y-slices (XZ planes at different y values)
        for y_val in np.linspace(y_min, y_max, 4):
            xx_2d, zz_2d = np.meshgrid(
                np.arange(x_min, x_max, h),
                np.arange(z_min, z_max, h)
            )
            
            # Points for this y-slice
            grid_points = np.zeros((xx_2d.size, 3))
            grid_points[:, 0] = xx_2d.ravel()
            grid_points[:, 1] = y_val
            grid_points[:, 2] = zz_2d.ravel()
            
            # Get predictions
            Z = model.predict(grid_points)
            Z = np.argmax(Z, axis=1).reshape(xx_2d.shape)
            
            # Plot each class region
            for i, color in zip(range(len(class_names)), colors):
                mask = Z == i
                if np.any(mask):
                    x_plot = xx_2d[mask]
                    z_plot = zz_2d[mask]
                    y_plot = np.full_like(x_plot, y_val)
                    
                    ax.scatter(x_plot, y_plot, z_plot, c=color, alpha=0.1, s=10)
        
        # Set labels and title
        combo_name = f"{combo[0]}-{combo[1]}-{combo[2]}"
        ax.set_title(f'3D Decision Boundary - Model {combo_name}')
        ax.set_xlabel(feature_one)
        ax.set_ylabel(feature_two)
        ax.set_zlabel(feature_three)
        ax.legend()
        
        # Adjust the viewing angle for better visualization
        ax.view_init(elev=30, azim=45)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(graficos_dir, f'decision_boundary_3d_{combo_name}.png'), dpi=300)
        
        # Create an animated version by rotating the view
        if hasattr(plt, 'animation'):  # Only if animation module is available
            try:
                from matplotlib.animation import FuncAnimation
                
                def rotate(angle):
                    ax.view_init(elev=30, azim=angle)
                    return fig,
                
                # Create animation with 36 frames (10 degrees per frame)
                ani = FuncAnimation(fig, rotate, frames=range(0, 360, 10), blit=True)
                ani.save(os.path.join(graficos_dir, f'decision_boundary_3d_animated_{combo_name}.gif'), 
                         writer='pillow', fps=5, dpi=150)
            except Exception as e:
                print(f"Could not create animation: {e}")


current_dir = os.path.dirname(os.path.abspath(__file__))
# Create graficos directory if it doesn't exist
graficos_dir = os.path.join(current_dir, 'graficos', 'modelos_tres_var')
os.makedirs(graficos_dir, exist_ok=True)

iris_df = pd.read_csv(os.path.join(current_dir, 'iris.csv'))

X = iris_df.drop(['Species','Id'], axis=1)
y = iris_df['Species']

# Get class names for confusion matrix
class_names = iris_df['Species'].unique()

# Create and train models for each feature combination
results = create_three_var_models(iris_df)

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