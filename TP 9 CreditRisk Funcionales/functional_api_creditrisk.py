import pandas as pd
import numpy as np
# from sklearn.preprocessing import OneHotEncoder # No longer needed here
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import regularizers
from datetime import datetime
import tensorflow.keras.backend as K

def plot_classification_report(report_dict, title='Classification Report', output_path='classification_report.png', class_names=None):
    """
    Generates and saves a classification report table as an image.

    Args:
        report_dict (dict): A dictionary like the one from sklearn.metrics.classification_report(output_dict=True).
        title (str): The title for the plot.
        output_path (str): The full path (including filename) to save the image.
        class_names (dict, optional): A dictionary mapping class labels (e.g., '0', '1') to display names (e.g., 'Red (0)', 'White (1)').
                                      If None, uses the keys from report_dict directly for class rows.
    """
    # Extract relevant data
    data = []
    row_labels = []

    # Process class-specific metrics
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict) and all(k in metrics for k in ['precision', 'recall', 'f1-score', 'support']):
            display_label = class_names.get(label, label) if class_names else label
            row_labels.append(display_label)
            data.append([
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1-score']:.3f}",
                f"{metrics['support']:.1f}"
            ])

    # Process overall metrics (accuracy, macro avg, weighted avg)
    # Ensure 'accuracy' is handled correctly as it's a single value, not a dict
    if 'accuracy' in report_dict:
        row_labels.append('accuracy')
        acc_value = report_dict['accuracy']
        # The example image has 'nan' for precision/recall for accuracy row, f1-score for accuracy, and support
        # We need to find where the overall support comes from, typically from macro/weighted avg.
        support_val = report_dict.get('macro avg', {}).get('support', report_dict.get('weighted avg', {}).get('support', np.nan))
        data.append([
            'nan',
            'nan',
            f"{acc_value:.3f}" if isinstance(acc_value, (int, float)) else acc_value, # accuracy is just one value
            f"{support_val:.1f}" if isinstance(support_val, (int, float)) else support_val
        ])

    for avg_type in ['macro avg', 'weighted avg']:
        if avg_type in report_dict:
            metrics = report_dict[avg_type]
            row_labels.append(avg_type)
            data.append([
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1-score']:.3f}",
                f"{metrics['support']:.1f}"
            ])

    col_labels = ['precision', 'recall', 'f1-score', 'support']

    if not data:
        print("No data to plot for classification report.")
        return

    fig, ax = plt.subplots(figsize=(8, len(row_labels) * 0.5 + 1))  # Adjust figsize based on number of rows
    ax.axis('tight')
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=data,
                     rowLabels=row_labels,
                     colLabels=col_labels,
                     cellLoc='center',
                     loc='center',
                     colColours=['#E0E0E0'] * len(col_labels)) # Light grey for column headers

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Adjust scale as needed

    # Style header cells
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor('#E0E0E0') # Header background
        if j == -1: # Row labels
             cell.set_text_props(weight='bold', color='black')


    plt.title(title, fontsize=16, pad=20)

    # Ensure directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Classification report saved to {output_path}")



# Create directory for saving plots and model artifacts
current_dir = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = os.path.join(current_dir, f'graficos/functional_api/{timestamp}/')
os.makedirs(model_dir, exist_ok=True)

# Load the preprocessed dataset
print("Loading the preprocessed dataset (clientes_final.csv)...")
try:
    final_df = pd.read_csv("clientes_final.csv") # Assuming clientes_final.csv is in the workspace root
except FileNotFoundError:
    print("Error: 'clientes_final.csv' not found. Make sure transformaciones.py has been run successfully.")
    exit()


# Display basic information about the dataset
print("\nDataset Info (from clientes_final.csv):")
print(final_df.info())
print("\nFirst few rows of the final DataFrame:")
print(final_df.head())

# --- Define Feature Split ---
# YOU NEED TO SET THIS VALUE based on your clientes_final.csv
# num_numerical_features_at_end should be the count of your original numerical columns
# that appear AFTER the one-hot encoded columns in clientes_final.csv (excluding 'Rechazo').
# For example, if 'credit-amount', 'tasa-interes', 'cantidad-creditos', 'cantidad-manutencion'
# are your numerical features and they are at the end, set this to 4.
# You can verify by looking at final_df.columns before X and y are split.
NUM_NUMERICAL_FEATURES_AT_END = 4 # <<<--- USER: SET THIS VALUE!
# Example: if transformaciones.py identifies these as numerical_columns_final:
# ['credit-amount', 'tasa-interes', 'cantidad-creditos', 'cantidad-manutencion']
# and they are indeed the last columns in X, then 4 is correct.

total_features = final_df.drop('Rechazo', axis=1).shape[1]
NUM_OHE_FEATURES_AT_START = total_features - NUM_NUMERICAL_FEATURES_AT_END

print(f"Total features: {total_features}")
print(f"Assuming OHE features at start: {NUM_OHE_FEATURES_AT_START}")
print(f"Assuming Numerical features at end: {NUM_NUMERICAL_FEATURES_AT_END}")


# --- Neural Network Implementation using Functional API ---
print("\nPreparing data for neural network training...")

# Separate features and target from the final DataFrame
if 'Rechazo' not in final_df.columns:
    print("Error: 'Rechazo' column not found in clientes_final.csv.")
    exit()
X = final_df.drop('Rechazo', axis=1)
y = final_df['Rechazo']

# Split the data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Split scaled data for multi-branch model
# OHE features are assumed to be at the start, numerical at the end
X_train_ohe = X_train_scaled[:, :NUM_OHE_FEATURES_AT_START]
X_train_num = X_train_scaled[:, NUM_OHE_FEATURES_AT_START:]

X_val_ohe = X_val_scaled[:, :NUM_OHE_FEATURES_AT_START]
X_val_num = X_val_scaled[:, NUM_OHE_FEATURES_AT_START:]

X_test_ohe = X_test_scaled[:, :NUM_OHE_FEATURES_AT_START]
X_test_num = X_test_scaled[:, NUM_OHE_FEATURES_AT_START:]

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,  # Reduced patience
    restore_best_weights=True
)

# Get input shape from the data
# input_shape = X_train_scaled.shape[1] # We'll pass dimensions directly now
# print(f"\nInput shape (number of features): {input_shape}")

def custom_cost_loss(y_true, y_pred_probs):
    """
    Custom binary cross-entropy loss function reflecting business costs.
    Cost of FN (y_true=1, pred_prob is low) = 5
    Cost of FP (y_true=0, pred_prob is high) = 1
    """
    y_true = K.cast(y_true, K.floatx())
    y_pred_probs = K.clip(y_pred_probs, K.epsilon(), 1 - K.epsilon()) # Clip to avoid log(0)

    cost_fn = 5.0  # Cost for misclassifying a true positive (bad customer as good)
    cost_fp = 1.0  # Cost for misclassifying a true negative (good customer as bad)

    loss = -K.mean( (cost_fn * y_true * K.log(y_pred_probs)) + \
                    (cost_fp * (1 - y_true) * K.log(1 - y_pred_probs)) )
    return loss

def build_model(dim_ohe, dim_numerical):
    """
    Build a multi-branch neural network model using Keras Functional API,
    inspired by the provided class example, now with Batch Normalization.
    
    Args:
        dim_ohe (int): Number of one-hot encoded input features.
        dim_numerical (int): Number of original numerical input features.
        
    Returns:
        Model: Compiled Keras model
    """
    # Input layers
    input_ohe = Input(shape=(dim_ohe,), name='input_ohe_features')
    input_numerical = Input(shape=(dim_numerical,), name='input_numerical_features')

    # Branch for OHE features
    dense_ohe = Dense(32, use_bias=False, name='dense_ohe_1')(input_ohe) # use_bias=False
    bn_ohe = BatchNormalization(name='bn_ohe_1')(dense_ohe)
    act_ohe = Activation('relu', name='act_ohe_1')(bn_ohe)
    processed_ohe = Dropout(0.3, name='dropout_ohe_1')(act_ohe)

    # Branch for numerical features
    dense_numerical = Dense(16, use_bias=False, name='dense_numerical_1')(input_numerical) # use_bias=False
    bn_numerical = BatchNormalization(name='bn_numerical_1')(dense_numerical)
    act_numerical = Activation('relu', name='act_numerical_1')(bn_numerical)
    processed_numerical = Dropout(0.2, name='dropout_numerical_1')(act_numerical)

    # Concatenate processed branches (ONLY processed features now)
    concatenated = Concatenate(name='concatenate_processed_only')([
        processed_ohe,
        processed_numerical
        # input_ohe and input_numerical have been removed from this list
    ])

    # Deeper processing on the combined features
    combined_dense1 = Dense(64, use_bias=False, kernel_regularizer=regularizers.l2(0.002), name='combined_dense_1')(concatenated)
    combined_bn1 = BatchNormalization(name='combined_bn_1')(combined_dense1)
    combined_act1 = Activation('relu', name='combined_act_1')(combined_bn1)
    combined_dropout1 = Dropout(0.4, name='dropout_combined_1')(combined_act1)
    
    combined_dense2 = Dense(32, use_bias=False, kernel_regularizer=regularizers.l2(0.002), name='combined_dense_2')(combined_dropout1)
    combined_bn2 = BatchNormalization(name='combined_bn_2')(combined_dense2)
    combined_act2 = Activation('relu', name='combined_act_2')(combined_bn2)
    combined_dropout2 = Dropout(0.3, name='dropout_combined_2')(combined_act2)


    # Output layer
    outputs = Dense(1, activation='sigmoid', name='output_layer')(combined_dropout2) # Input is now from combined_dropout_2
    
    # Create the model
    model = Model(inputs=[input_ohe, input_numerical], outputs=outputs, name='credit_risk_custom_cost_loss_model') 
    
    return model

# Create and compile the model
print("\nBuilding and compiling the multi-branch model...")
model = build_model(NUM_OHE_FEATURES_AT_START, NUM_NUMERICAL_FEATURES_AT_END)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=custom_cost_loss,  # Use the custom loss function
    metrics=['accuracy']  # Track accuracy during training
)

# Try to generate and save model architecture plot
try:
    plot_model(
        model,
        to_file=os.path.join(model_dir, 'model_plot.png'),
        show_shapes=True,
        show_layer_names=True
    )
    print("\nModel architecture plot has been saved to model_outputs/model_plot.png")
except ImportError:
    print("\nWarning: Could not generate model plot. Please install pydot and graphviz:")
    print("pip install pydot graphviz")
except Exception as e:
    print(f"\nWarning: Could not generate model plot due to an error: {e}")


# Print model summary
model.summary()

# Train the model
print("\nTraining the model...")
history = model.fit(
    [X_train_ohe, X_train_num], # Pass inputs as a list
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=([X_val_ohe, X_val_num], y_val), # Pass validation inputs as a list
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model on test data
print("\nEvaluating the model on test data...")
test_loss, test_accuracy = model.evaluate([X_test_ohe, X_test_num], y_test, verbose=0) # Pass test inputs as a list
print(f"Test accuracy: {test_accuracy:.4f}")

# Make predictions
y_pred_prob = model.predict([X_test_ohe, X_test_num]) # Pass test inputs as a list

# ---- START THRESHOLD ITERATION ----
print("\n--- Evaluating different decision thresholds to minimize business cost ---")
thresholds_to_try = np.arange(0.05, 0.96, 0.05).tolist() # Adjusted range for wider search

min_overall_cost = float('inf') # Initialize with a very high cost
best_threshold_for_min_cost = -1
best_report_dict_for_min_cost = None
best_y_pred_int_for_min_cost_cm = None
all_threshold_metrics = []

# Ensure y_test is a NumPy array for direct indexing if it's a pandas Series
y_test_np = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test

custom_class_names_nn = {
    '0': 'No Rechazo (0)',
    '1': 'Rechazo (1)'
}

COST_FN = 5
COST_FP = 1

for t in thresholds_to_try:
    current_threshold = round(t, 2)
    print(f"\n--- Threshold: {current_threshold:.2f} ---")
    y_pred_at_threshold = (y_pred_prob.squeeze() > current_threshold).astype(int)

    # Calculate Business Cost
    fn_count = 0
    fp_count = 0
    for i in range(len(y_test_np)):
        if y_test_np[i] == 1 and y_pred_at_threshold[i] == 0: # Actual 1, Predicted 0 -> FN
            fn_count += 1
        elif y_test_np[i] == 0 and y_pred_at_threshold[i] == 1: # Actual 0, Predicted 1 -> FP
            fp_count += 1
    current_business_cost = (fn_count * COST_FN) + (fp_count * COST_FP)
    print(f"Business Cost: {current_business_cost} (FN: {fn_count}, FP: {fp_count})")

    report_text = classification_report(y_test_np, y_pred_at_threshold, zero_division=0, target_names=['No Rechazo (0)', 'Rechazo (1)'])
    print(report_text)
    report_dict_at_threshold = classification_report(y_test_np, y_pred_at_threshold, output_dict=True, zero_division=0, target_names=['No Rechazo (0)', 'Rechazo (1)'])

    # ---- START: Collect metrics for heatmap ----
    metrics_for_heatmap = {'threshold': current_threshold}
    metrics_for_heatmap['business_cost'] = current_business_cost # Add business cost
    metrics_for_heatmap['accuracy'] = report_dict_at_threshold.get('accuracy', 0)
    for class_label_key in ['0', '1']: # Use string keys '0' and '1'
        display_name = custom_class_names_nn[class_label_key] # e.g., 'No Rechazo (0)'
        class_metrics = report_dict_at_threshold.get(display_name, {})

        # Suffix for heatmap column names
        suffix = display_name.replace(' ', '_').replace('(', '').replace(')', '') # e.g., No_Rechazo_0

        metrics_for_heatmap[f'precision_{suffix}'] = class_metrics.get('precision', 0)
        metrics_for_heatmap[f'recall_{suffix}'] = class_metrics.get('recall', 0)
        metrics_for_heatmap[f'f1-score_{suffix}'] = class_metrics.get('f1-score', 0)
    all_threshold_metrics.append(metrics_for_heatmap)
    # ---- END: Collect metrics for heatmap ----

    if current_business_cost < min_overall_cost:
        min_overall_cost = current_business_cost
        best_threshold_for_min_cost = current_threshold
        best_report_dict_for_min_cost = report_dict_at_threshold
        best_y_pred_int_for_min_cost_cm = y_pred_at_threshold
        print(f"*** New best threshold found: {best_threshold_for_min_cost:.2f} with cost: {min_overall_cost} ***")


# --- Output and save plot for the best threshold based on MINIMUM BUSINESS COST ---
if best_report_dict_for_min_cost:
    print(f"\n--- Best Threshold for Minimum Business Cost: {best_threshold_for_min_cost:.2f} (Cost: {min_overall_cost}) ---")
    if best_y_pred_int_for_min_cost_cm is not None:
         print(classification_report(y_test_np, best_y_pred_int_for_min_cost_cm, zero_division=0, target_names=['No Rechazo (0)', 'Rechazo (1)']))

    plot_classification_report(
        best_report_dict_for_min_cost,
        title=f'Best CR NN (Min Cost {min_overall_cost}) - Thr {best_threshold_for_min_cost:.2f}',
        output_path=os.path.join(model_dir, 'classification_report_min_cost.png'),
        class_names=custom_class_names_nn
    )
    print(f"Saved min cost classification report table to {os.path.join(model_dir, 'classification_report_min_cost.png')}")

    if best_y_pred_int_for_min_cost_cm is not None:
        cm_min_cost = confusion_matrix(y_test_np, best_y_pred_int_for_min_cost_cm)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_min_cost, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Rechazo (0)', 'Rechazo (1)'],
                    yticklabels=['No Rechazo (0)', 'Rechazo (1)'])
        plt.title(f'CM - Min Cost {min_overall_cost} (Thr {best_threshold_for_min_cost:.2f})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(model_dir, 'confusion_matrix_min_cost.png'))
        plt.close()
        print(f"Saved min cost confusion matrix to {os.path.join(model_dir, 'confusion_matrix_min_cost.png')}")
else:
    print("\nCould not determine best threshold for minimum business cost based on the tested range.")

# ---- END THRESHOLD ITERATION ----

# ---- START: Generate heatmap of threshold performances ----
if all_threshold_metrics:
    metrics_df = pd.DataFrame(all_threshold_metrics)
    metrics_df = metrics_df.set_index('threshold')

    # Separate DataFrames for different scales
    cost_df = metrics_df[['business_cost']]
    other_metrics_df = metrics_df.drop(columns=['business_cost'])

    # Plot for Business Cost
    plt.figure(figsize=(6, 8)) # Adjusted for a single column
    sns.heatmap(cost_df, annot=True, fmt=".0f", cmap="viridis_r", linewidths=.5, cbar_kws={'label': 'Business Cost'}) # fmt=".0f" for integer cost
    plt.title('Business Cost vs. Decision Threshold')
    plt.ylabel('Threshold')
    plt.xlabel('') # No x-label needed for single column
    plt.xticks([]) # Hide x-axis ticks
    plt.tight_layout()
    cost_heatmap_filename = 'threshold_performance_heatmap_cost.png'
    plt.savefig(os.path.join(model_dir, cost_heatmap_filename))
    plt.close()
    print(f"\nSaved business cost heatmap to: {os.path.join(model_dir, cost_heatmap_filename)}")

    # Plot for Other Metrics
    if not other_metrics_df.empty:
        plt.figure(figsize=(12, 8)) # Adjusted size as needed
        sns.heatmap(other_metrics_df, annot=True, fmt=".2f", cmap="viridis", linewidths=.5, vmin=0, vmax=1, cbar_kws={'label': 'Metric Value (0-1)'}) # fmt=".2f", set vmin/vmax for 0-1 scale
        plt.title('Performance Metrics (0-1 Scale) vs. Decision Threshold')
        plt.ylabel('Threshold')
        plt.xlabel('Metrics')
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        other_metrics_heatmap_filename = 'threshold_performance_heatmap_metrics.png'
        plt.savefig(os.path.join(model_dir, other_metrics_heatmap_filename))
        plt.close()
        print(f"Saved other metrics heatmap to: {os.path.join(model_dir, other_metrics_heatmap_filename)}")
    else:
        print("\nNo other metrics (besides business cost) to plot in heatmap.")

else:
    print("\nNo metrics collected for threshold performance heatmap.")
# ---- END: Generate heatmap of threshold performances ----

# --- Training History Plot (Remains the same) ---
if 'history' in locals() and history is not None:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    history_plot_filename = 'training_history.png'
    plt.savefig(os.path.join(model_dir, history_plot_filename))
    plt.close()
    print(f"\nTraining history plot saved to: {os.path.join(model_dir, history_plot_filename)}")
else:
    print("\nTraining history (history object) not available. Skipping history plot.")

# --- Model Plot (Remains the same, saved earlier) ---
model_plot_path = os.path.join(model_dir, 'model_plot.png')
if os.path.exists(model_plot_path):
    print(f"Model architecture plot is available at: {model_plot_path}")
else:
    print(f"Model architecture plot was attempted to be saved at: {model_plot_path} (check earlier warnings if it's missing).")


print(f"\nAll requested artifacts are saved in the directory: {model_dir}")
print("\n--- End of Script ---")
