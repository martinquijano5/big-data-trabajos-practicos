import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import datetime



# Activation functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def create_model_prediction(row, feature_cols):
    """
    Makes a prediction for a single row using the predefined neural network.
    """
    # Extract features for the current row (assumes they are already scaled)
    x_input = row[feature_cols].values.astype(float)

    # Layer 0 calculation (Hidden Layer with ReLU)
    # x_input shape: (4,), w_l0 shape: (4, 2) -> z_l0 shape: (2,)
    z_l0 = np.dot(x_input, w_l0) + b_l0                             # equivalente a mmult de excel
    a_l0 = relu(z_l0)  # Output of Hidden Layer, shape: (2,)        equivalente al max de excel

    # Layer 1 calculation (Output Layer with Sigmoid)
    # a_l0 shape: (2,), w_l1 shape: (2, 1) -> z_l1 shape: (1,)
    z_l1 = np.dot(a_l0, w_l1) + b_l1                                # equivalente a mmult de excel
    output_sigmoid = sigmoid(z_l1)  # Output of Sigmoid neuron, shape: (1,) 

    # convierto la salida sigmoide a 0 o 1 
    predicted_type = 1 if output_sigmoid[0] >= 0.5 else 0
    return predicted_type

def save_predictions_to_csv(df_with_predictions, original_combined_df, base_dir, target_col_name='real_type', predicted_col_name='predicted_type', original_feature_cols=None):
    """
    Saves the predictions to CSV files, separating white and red wines.
    Uses original (unscaled) feature values.
    """
    # Create a DataFrame for export with original unscaled features and types
    export_df = original_combined_df.reset_index(drop=True) # Ensure RangeIndex for alignment

    # Add the predicted_type column. df_with_predictions (scaledData) also has a RangeIndex.
    export_df[predicted_col_name] = df_with_predictions[predicted_col_name].values

    # Select relevant columns for final CSV (original features + real_type + predicted_type)
    if original_feature_cols is None: # Should be all columns from original_combined_df except real_type if it was added there
        original_feature_cols = [col for col in original_combined_df.columns if col != target_col_name]

    columns_to_save = original_feature_cols + [target_col_name, predicted_col_name]
    # Ensure all columns exist in export_df
    columns_to_save = [col for col in columns_to_save if col in export_df.columns]


    white_wine_pred_df = export_df[export_df[target_col_name] == 1][columns_to_save]
    red_wine_pred_df = export_df[export_df[target_col_name] == 0][columns_to_save]

    white_csv_path = os.path.join(base_dir, "white_wine_predictions.csv")
    red_csv_path = os.path.join(base_dir, "red_wine_predictions.csv")

    white_wine_pred_df.to_csv(white_csv_path, index=False)
    red_wine_pred_df.to_csv(red_csv_path, index=False)
    print(f"Saved white wine predictions (with original features) to {white_csv_path}")
    print(f"Saved red wine predictions (with original features) to {red_csv_path}")



#crear directorios
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, 'generate_color_csv')
os.makedirs(parent_dir, exist_ok=True)

#importar los csv originales
white_wine_data = pd.read_csv("sommelier secuenciales/winequality-white.csv", sep=';')
red_wine_data = pd.read_csv("sommelier secuenciales/winequality-red.csv", sep=';')


white_wine_data['real_type'] = 1 # 1 for white
red_wine_data['real_type'] = 0 # 0 for red
combined_data = pd.concat([white_wine_data, red_wine_data], axis=0)

#defino las columnas que voy a usar
feature_columns = ['total sulfur dioxide', 'chlorides', 'volatile acidity', 'residual sugar']
target_column = 'real_type'

#escalo los datos (solo las x)
X = combined_data.drop(columns=[target_column])
y = combined_data[target_column]
scaledData = StandardScaler().fit_transform(X)
scaledData = pd.DataFrame(scaledData, columns=X.columns)

# Add the target variable to scaledData
scaledData[target_column] = y.values

print(scaledData.head())
print(scaledData.tail())


#generar el modelo en base a la estructura, weights y biases
#neural network with one hidden layer (relu activation function), 2 neurons, 1 output neuron (sigmoid activation function)
# Weights and Biases from the provided images
# Layer 0: InputFeat to L0 Weights (Inputs: 'total sulfur dioxide', 'chlorides', 'volatile acidity', 'residual sugar')
w_l0 = np.array([
    [-0.3974, -2.0489],  # weights for total sulfur dioxide to L0_N1, L0_N2
    [-3.5143, -0.0800],  # weights for chlorides to L0_N1, L0_N2
    [ 1.0945,  1.5660],  # weights for volatile acidity to L0_N1, L0_N2
    [-0.0647, -0.5882]   # weights for residual sugar to L0_N1, L0_N2
])
# Layer 0: Keras Layer 0 Biases
b_l0 = np.array([0.8428, 1.3576]) # L0_N1 Bias, L0_N2 Bias

# Layer 1: L0_Out to L1 Weights
w_l1 = np.array([
    [1.9710],   # weight from L0_N1 to L1_N1
    [-2.1706]   # weight from L0_N2 to L1_N1
])
# Layer 1: Keras Layer 1 Biases
b_l1 = np.array([4.6380]) # L1_N1 Bias

#hacer que el modelo haga predicciones para cada dato
print("\nGenerating predictions...")
scaledData['predicted_type'] = scaledData.apply(
    lambda row: create_model_prediction(row, feature_columns),
    axis=1
)

#hacer que el modelo haga predicciones para cada dato
print("\nscaledData with predicted_type (head):")
print(scaledData[[*feature_columns, 'real_type', 'predicted_type']].head())
print("\nscaledData with predicted_type (tail):")
print(scaledData[[*feature_columns, 'real_type', 'predicted_type']].tail())

# Save the results to CSV files
# The combined_data DataFrame has the original unscaled features.
# feature_columns contains the names of features used in the model
# X.columns contains all original feature names before scaling.
save_predictions_to_csv(
    scaledData,
    combined_data,
    parent_dir,
    target_col_name=target_column,
    predicted_col_name='predicted_type',
    original_feature_cols=list(X.columns) # Pass all original feature column names
)

