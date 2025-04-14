import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


# Set up directories
current_dir = os.path.dirname(os.path.abspath(__file__))


# Load data
df = pd.read_csv('TP 7 Churn RNS/Iran Customer Churn.csv')
df = df.drop(['Status','Age', 'Age Group', 'Charge Amount'], axis=1)  # Remove Status variable


# Define categorical columns that should not be normalized
categorical_cols = ['Complains', 'Tariff Plan', 'Churn']
numerical_cols = df.columns.difference(categorical_cols).tolist()


# Scale only the numerical columns
scaler = StandardScaler()
df_numerical_scaled = pd.DataFrame(
    scaler.fit_transform(df[numerical_cols]),
    columns=numerical_cols
)

# Combine the original categorical columns with the scaled numerical columns
df_scaled = pd.concat([df[categorical_cols], df_numerical_scaled], axis=1)

print(df_scaled.head())
print(df_scaled.describe())
print(df_scaled.info())

# Save the scaled data to a CSV file
df_scaled.to_excel('TP 7 Churn RNS/Iran Customer Churn_scaled.xlsx', index=False)

# Add model creation and training functions
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
    categorical_cols = ['Complains', 'Tariff Plan']
    numerical_cols = [col for col in df.columns if col != 'Churn' and col not in categorical_cols]
    
    # Create a list that represents the order of features in X_scaled
    feature_cols = categorical_cols + numerical_cols
    
    X_categorical = df[categorical_cols].values
    X_numerical = df[numerical_cols].values
    
    # Use already scaled data
    X_numerical_scaled = X_numerical
    
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

# Train model with the scaled data
print("\nTraining neural network model with the scaled data...")
results = train_complete_model(df_scaled)
print(f"Model Accuracy: {results['accuracy']:.4f}")

# Export model weights
print("\nExporting model weights to Excel...")
print_model_weights(results, 3)