import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score

def full_model(shape):
    model = Sequential([
        Dense(16, activation='relu', input_shape=(shape,), kernel_regularizer=l1(0.03)),
        Dense(8, activation='relu', kernel_regularizer=l1(0.03)),
        Dense(1, activation='sigmoid')  # Binary classification (Churn vs No Churn)
    ])
    
    optimizer = Adam(learning_rate=0.01)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_complete_model(df):
    # Define categorical and numerical columns
    categorical_cols = ['Complains', 'Charge  Amount', 'Tariff Plan']
    numerical_cols = [col for col in df.columns if col != 'Churn' and col not in categorical_cols]
    
    # Create a list that represents the order of features in X_scaled
    feature_cols = categorical_cols + numerical_cols
    
    X_categorical = df[categorical_cols].values
    X_numerical = df[numerical_cols].values
    
    # Scale the data
    scaler = StandardScaler()
    X_numerical_scaled = scaler.fit_transform(X_numerical)
    
    # Concatenate the variables
    X_scaled = np.concatenate([X_categorical, X_numerical_scaled], axis=1)
    
    y = df['Churn'].values

    # Create and train model
    input_shape = X_scaled.shape[1]
    model = full_model(input_shape)
    history = model.fit(
        X_scaled, y,
        epochs=300,
        batch_size=32,
        verbose=1,
        validation_split=0.2
    )
    
    # Evaluate on the training data
    loss, accuracy = model.evaluate(X_scaled, y, verbose=1)
    
    # Get predictions
    y_pred_proba = model.predict(X_scaled)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Store results
    results = {
        'model': model,
        'history': history,
        'accuracy': accuracy,
        'loss': loss,
        'X_scaled': X_scaled,
        'y': y,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'feature_names': feature_cols
    }
    
    return results

def plot_training_results(results):
    # Create separate figure for loss
    plt.figure(figsize=(12, 8))
    plt.title('Pérdida durante el entrenamiento', fontsize=14)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Pérdida', fontsize=12)
    plt.grid(True)
    
    plt.plot(results['history'].history['loss'], color='blue', label='Train loss')
    plt.plot(results['history'].history['val_loss'], color='blue', 
            alpha=0.5, linestyle='--', label='Val loss')
    
    plt.ylim(0, 1)  # Set y-axis from 0 to 1
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graficos_dir, 'loss_completo.png'), dpi=300)
    
    # Create separate figure for accuracy
    plt.figure(figsize=(12, 8))
    plt.title('Precisión durante el entrenamiento', fontsize=14)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Precisión', fontsize=12)
    plt.grid(True)
    
    plt.plot(results['history'].history['accuracy'], color='green', label='Train accuracy')
    plt.plot(results['history'].history['val_accuracy'], color='green', 
            alpha=0.5, linestyle='--', label='Val accuracy')
    
    plt.ylim(0, 1)  # Set y-axis from 0 to 1
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graficos_dir, 'accuracy_completo.png'), dpi=300)

def plot_model_accuracy(results, output_filename='model_metrics.png'):
    y_true = results['y']
    y_pred = results['y_pred']
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')  # Hide axes
    ax.axis('tight')  # Tight layout
    
    # Create data for table
    metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    values = [f'{accuracy:.4f}', f'{f1:.4f}', f'{precision:.4f}', f'{recall:.4f}']
    
    # Create table with colored cells based on value
    cell_colors = []
    for val in [accuracy, f1, precision, recall]:
        # Color gradient from red (0.5) to green (1.0)
        if val < 0.7:
            color = (1.0, 0.7, 0.7)  # light red
        elif val < 0.8:
            color = (1.0, 0.9, 0.7)  # yellow
        elif val < 0.9:
            color = (0.8, 1.0, 0.8)  # light green  
        else:
            color = (0.6, 1.0, 0.6)  # green
        cell_colors.append(color)
    
    table = ax.table(
        cellText=[values],
        colLabels=metrics,
        loc='center',
        cellLoc='center',
        cellColours=[cell_colors]
    )
    
    # Customize table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)  # Adjust table size
    
    plt.title('Métricas del Modelo', fontsize=14, pad=20)
    plt.tight_layout()
    
    # Save figure with the specified filename
    plt.savefig(os.path.join(graficos_dir, output_filename), dpi=300)

def plot_confusion_matrix(results):
    y = results['y']
    y_pred = results['y_pred']
    
    # Calculate confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Create figure and plot
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                 display_labels=['No Churn', 'Churn'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    
    plt.title('Matriz de Confusión - Modelo Completo')
    plt.tight_layout()
    plt.savefig(os.path.join(graficos_dir, 'confusion_matrix_completo.png'), dpi=300)

def plot_feature_importance(results):
    model = results['model']
    feature_names = results['feature_names']
    
    # Get weights from the first layer
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
    
    # Create horizontal bar chart
    bars = plt.barh(range(len(indices)), importance[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[indices[i]] for i in range(len(indices))])
    
    # Add text labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', va='center')
    
    plt.xlabel('Importancia relativa')
    plt.ylabel('Características')
    plt.tight_layout()
    plt.savefig(os.path.join(graficos_dir, 'feature_importance.png'), dpi=300)

def sensibility_analisis(results):
    """
    Perform a sensitivity analysis by iteratively removing the least important 
    features and tracking model performance.
    """
    original_X = results['X_scaled']
    y = results['y']
    feature_names = results['feature_names'].copy()
    
    # Initialize lists to track results
    accuracies = []
    features_remaining = []
    removed_features = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    models = []  # Store all models
    
    # Store initial results
    initial_accuracy = results['accuracy']
    initial_f1 = f1_score(y, results['y_pred'])
    initial_precision = precision_score(y, results['y_pred'])
    initial_recall = recall_score(y, results['y_pred'])
    
    accuracies.append(initial_accuracy)
    f1_scores.append(initial_f1)
    precision_scores.append(initial_precision)
    recall_scores.append(initial_recall)
    features_remaining.append(len(feature_names))
    models.append(results['model'])  # Add the original model
    
    # Track best model
    best_accuracy = initial_accuracy
    best_model = results['model']
    best_model_features = feature_names.copy()
    best_model_index = 0
    
    # Keep a copy of the current features and X data
    current_features = feature_names.copy()
    current_X = original_X.copy()
    
    print("\n== SENSITIVITY ANALYSIS ==")
    print(f"Starting with {len(feature_names)} features. Accuracy: {initial_accuracy:.4f}, F1: {initial_f1:.4f}, Precision: {initial_precision:.4f}, Recall: {initial_recall:.4f}")
    
    # Use initial model for first iteration
    model = results['model']
    
    # Continue until only two features remain
    while len(current_features) > 2:
        # Get feature importances from current model
        weights = model.layers[0].get_weights()[0]
        importance = np.sum(np.abs(weights), axis=1)
        importance_normalized = importance / np.sum(importance)
        
        # Find least important feature
        min_idx = np.argmin(importance_normalized)
        least_important_feature = current_features[min_idx]
        removed_features.append(least_important_feature)
        
        print(f"Removing feature: {least_important_feature} (importance: {importance_normalized[min_idx]:.4f})")
        
        # Remove the least important feature
        current_features.pop(min_idx)
        current_X = np.delete(current_X, min_idx, axis=1)
        
        # Create a new model with the updated input shape
        input_shape = current_X.shape[1]
        model = full_model(input_shape)
        model.fit(
            current_X, y,
            epochs=100,
            batch_size=32,
            verbose=1,
            validation_split=0.2
        )
        
        # Evaluate new model
        y_pred_proba = model.predict(current_X, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        
        # Store results
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precision_scores.append(prec)
        recall_scores.append(rec)
        features_remaining.append(len(current_features))
        models.append(model)
        
        # Check if this model is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_features = current_features.copy()
            best_model_index = len(models) - 1
        
        print(f"Features remaining: {len(current_features)}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
    
    # Create a list for x-axis labels with initial state and then removed features
    x_labels = ['Original'] + removed_features
    
    # Create a figure for the timeline view
    plt.figure(figsize=(14, 8))
    plt.title('Evolución de la accuracy al eliminar características', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy - including initial accuracy and all iterations
    plt.plot(range(len(accuracies)), accuracies, marker='o', linestyle='-', color='#3498db', linewidth=2, label='Accuracy')
    
    # Highlight best model
    plt.plot(best_model_index, best_accuracy, 'ro', markersize=10, label='Best Model')
    
    # Set x-axis labels to show features removed at each step
    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha='right')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graficos_dir, 'sensitivity_analysis.png'), dpi=300)
    
    # Print the final two most important features
    print("\nFinal two most important features:")
    print(f"1. {current_features[0]}")
    print(f"2. {current_features[1]}")
    
    # Print information about the best model
    print(f"\nBest model had {len(best_model_features)} features with accuracy: {best_accuracy:.4f}")
    print(f"Best model features: {', '.join(best_model_features)}")
    
    # Save the best model metrics as an image
    # If the best model is not the original model, we need to generate its predictions
    if best_model_index > 0:
        # Get all removed features up to the best model index
        features_to_remove = removed_features[:best_model_index]
        
        # Start with a copy of the original data and features
        best_X = original_X.copy()
        best_features = feature_names.copy()
        
        # Remove each feature one by one
        for feature_to_remove in features_to_remove:
            idx_to_remove = best_features.index(feature_to_remove)
            best_X = np.delete(best_X, idx_to_remove, axis=1)
            best_features.remove(feature_to_remove)
        
        # Verify we have the correct number of features
        assert len(best_features) == len(best_model_features), f"Feature count mismatch: {len(best_features)} vs expected {len(best_model_features)}"
        
        # Get best model's predictions
        y_pred_best = (best_model.predict(best_X, verbose=0) > 0.5).astype(int)
        
        # Create results dictionary for the best model
        best_results = {
            'y': y,
            'y_pred': y_pred_best
        }
        
        # Save best model metrics
        plot_model_accuracy(best_results, 'best_model_metrics@model_metrics.png')
    else:
        # If the best model is the original model, just use the original results
        plot_model_accuracy(results, 'best_model_metrics@model_metrics.png')
    
    # Return the results for further analysis if needed
    return {
        'accuracies': accuracies,
        'f1_scores': f1_scores,
        'precision_scores': precision_scores,
        'recall_scores': recall_scores,
        'features_remaining': features_remaining,
        'removed_features': removed_features,
        'most_important_features': current_features,
        'best_model': best_model,
        'best_model_features': best_model_features,
        'best_accuracy': best_accuracy
    }

def print_model_weights(results, num_layers):
    model = results['model']
    
    # Create a dictionary to store all weights and biases for Excel export
    excel_data = {}
    
    for i in range(num_layers):
        layer_weights = model.layers[i].get_weights()[0]
        layer_biases = model.layers[i].get_weights()[1]
        
        print(f"\nLayer {i+1} Weights:")
        print(layer_weights)
        print(f"\nLayer {i+1} Biases:")
        print(layer_biases)
        
        # Add to excel data dictionary
        excel_data[f'Layer_{i+1}_Weights'] = layer_weights
        excel_data[f'Layer_{i+1}_Biases'] = layer_biases
    
    # Export to Excel using pandas ExcelWriter with openpyxl engine
    # Make sure openpyxl is installed: pip install openpyxl
    writer = pd.ExcelWriter(os.path.join(current_dir, 'model_weights.xlsx'), engine='openpyxl')
    
    # Create DataFrames for each layer and write to separate sheets
    for i in range(num_layers):
        # Weights
        weights = excel_data[f'Layer_{i+1}_Weights']
        if i == 0:
            # For first layer, use feature names as indices
            weights_df = pd.DataFrame(
                weights, 
                index=results['feature_names'],
                columns=[f'Neuron_{j+1}' for j in range(weights.shape[1])]
            )
        else:
            # For other layers, use generic indices
            weights_df = pd.DataFrame(
                weights,
                columns=[f'Neuron_{j+1}' for j in range(weights.shape[1])]
            )
        
        # Biases
        biases = excel_data[f'Layer_{i+1}_Biases']
        biases_df = pd.DataFrame(biases, columns=['Bias'])
        
        # Write to Excel
        weights_df.to_excel(writer, sheet_name=f'Layer_{i+1}_Weights')
        biases_df.to_excel(writer, sheet_name=f'Layer_{i+1}_Biases')
    
    writer.close()
    print(f"\nModel weights and biases exported to: {os.path.join(current_dir, 'model_weights.xlsx')}")

def create_simple_model(df, simple_features, output_filename='simple_model_metrics.png'):
    print("\nCreating a simple model with the two most important features...")
    print(f"Using features: {simple_features}")

    # Use unscaled data directly
    X_simple = df[simple_features].values
    y = df['Churn'].values

    # Create and train the simple model
    simple_model = full_model(X_simple.shape[1])
    simple_history = simple_model.fit(
        X_simple, y,
        epochs=300,
        batch_size=32,
        verbose=1,
        validation_split=0.2
    )

    # Get predictions
    y_pred_proba = simple_model.predict(X_simple)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Store results
    simple_results = {
        'y': y,
        'y_pred': y_pred
    }

    # Create the metrics table
    plot_model_accuracy(simple_results, output_filename)
    
    # Return both results and the model
    return simple_results, simple_model

def plot_decision_boundary(df, simple_features, simple_model):
    print("\nCreating decision boundary plot for the two-variable model...")
    
    # Extract features directly without scaling
    X = df[simple_features].values
    y = df['Churn'].values
    
    # Create a mesh grid
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh grid
    Z = simple_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = (Z > 0.5).astype(int)
    Z = Z.reshape(xx.shape)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot decision boundary with distinct colors for each class
    # Use a binary colormap with only two colors (not a gradient)
    plt.contourf(xx, yy, Z, alpha=0.6, levels=1, colors=['peachpuff', 'lightblue'])
    
    # Plot data points
    markers = ['o', '^']
    colors = ['maroon', 'navy']
    labels = ['No Churn', 'Churn']
    
    for i, label in enumerate([0, 1]):
        plt.scatter(X[y == label, 0], X[y == label, 1], 
                   c=colors[i], marker=markers[i], 
                   edgecolors='k', s=30, label=labels[i])
    
    # Add feature names as axis labels
    plt.xlabel(f"Frequency of use", fontsize=12)
    plt.ylabel(f"Frequency of SMS", fontsize=12)
    
    # Add legend
    plt.legend(title="Churn")
    
    plt.title("Límite de Decisión - Modelo con Dos Variables", fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(graficos_dir, 'decision_boundary.png'), dpi=300)
    
    return plt

# Set up directories
current_dir = os.path.dirname(os.path.abspath(__file__))
graficos_dir = os.path.join(current_dir, 'graficos', 'modelo_completo')
os.makedirs(graficos_dir, exist_ok=True)

# Load data
df = pd.read_csv('TP 7 Churn RNS/Iran Customer Churn.csv')
df = df.drop(['Status','Age', 'Age Group'], axis=1)  # Remove Status, age and age group variables
df['Tariff Plan'] = df['Tariff Plan'].replace({2: 1, 1: 0})


# Create and train the model
results = train_complete_model(df)

# Print results
print(f"Model Accuracy: {results['accuracy']:.4f}")

# Plot results
plot_training_results(results)
plot_confusion_matrix(results)
plot_feature_importance(results)
plot_model_accuracy(results)


# Get the sensitivity analysis results
sensitivity_results = sensibility_analisis(results)

# Store the best model
best_model = sensitivity_results['best_model']
most_important_features = sensitivity_results['most_important_features']
best_model_features = sensitivity_results['best_model_features']
best_accuracy = sensitivity_results['best_accuracy']

print(f"Best model: {best_model}")
print(f"Most important features: {most_important_features}")
print(f"Best model features: {best_model_features}")
print(f"Best accuracy: {best_accuracy}")

# Print model weights and biases
print_model_weights(results, 3)

# Create a simple model with only the two most important features
simple_results, simple_model = create_simple_model(df, sensitivity_results['most_important_features'])

# Plot decision boundary with unscaled data and distinct colors
plot_decision_boundary(df, sensitivity_results['most_important_features'], simple_model)

# Show plots
plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')