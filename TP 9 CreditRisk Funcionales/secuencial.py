import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
# from tensorflow.keras.utils import plot_model # No longer needed

# Define output directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODELS_DIR = os.path.join(BASE_DIR, 'graficos', 'secuencial', 'models') # No longer needed
PLOTS_DIR = os.path.join(BASE_DIR, 'graficos', 'secuencial', 'plots')
# RESULTS_DIR = os.path.join(BASE_DIR, 'graficos', 'secuencial', 'results') # No longer needed

# Create necessary directories
# for directory in [MODELS_DIR, PLOTS_DIR, RESULTS_DIR]: # Modified
#     os.makedirs(directory, exist_ok=True)
#     print(f"Created directory: {directory}")
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"Created directory: {PLOTS_DIR}")


# 1. Load the data
df = pd.read_csv('clientes_final.csv')

# 2. Split into features (X) and target (y)
X = df.drop('Rechazo', axis=1)
y = df['Rechazo']

# 3. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler parameters for future use - REMOVED
# scaler_params = {
# 'mean_': scaler.mean_,
# 'scale_': scaler.scale_
# }
# np.save(os.path.join(MODELS_DIR, 'scaler_params.npy'), scaler_params)
# print(f"Saved scaler parameters to: {os.path.join(MODELS_DIR, 'scaler_params.npy')}")

# 4. Split into train (60%), val (20%), test (20%) with stratification
def split_data(X, y):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_scaled, y)

# Save the split data for future reference - REMOVED
# split_data = {
# 'X_train': X_train,
# 'X_val': X_val,
# 'X_test': X_test,
# 'y_train': y_train,
# 'y_val': y_val,
# 'y_test': y_test
# }
# np.save(os.path.join(RESULTS_DIR, 'split_data.npy'), split_data)
# print(f"Saved split data to: {os.path.join(RESULTS_DIR, 'split_data.npy')}")

# 5. Define class weights to handle imbalance
#    Class 1 (Rechazo) is more costly for the business if missed, so we upweight it.
class_weights = {0: 1, 1: 5}

# 6. Build the Sequential model
model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),  # Input layer
    keras.layers.Dense(64, activation='relu'),      # Hidden layer 1
    keras.layers.Dense(32, activation='relu'),      # Hidden layer 2
    keras.layers.Dense(1, activation='sigmoid')     # Output layer
])

# 7. Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 8. Train the model with class weights
#    Class weights help the model pay more attention to the minority class (Rechazo=1),
#    reducing the risk of costly false negatives for the business.
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

# Save the trained model - REMOVED
# model_save_path = os.path.join(MODELS_DIR, 'credit_risk_model.h5')
# model.save(model_save_path)
# print(f"Saved trained model to: {model_save_path}")

# Save training history - REMOVED .npy, plot is kept
# history_dict = history.history
# np.save(os.path.join(RESULTS_DIR, 'training_history.npy'), history_dict)
# print(f"Saved training history to: {os.path.join(RESULTS_DIR, 'training_history.npy')}")

# 9. Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_acc:.4f}")

# Save test results - REMOVED
# test_results = {
# 'test_loss': test_loss,
# 'test_accuracy': test_acc
# }
# np.save(os.path.join(RESULTS_DIR, 'test_results.npy'), test_results)
# print(f"Saved test results to: {os.path.join(RESULTS_DIR, 'test_results.npy')}")

# 10. Generate and save evaluation plots (Confusion Matrix and Classification Report Table)
def generate_evaluation_plots(model, X, y_true, chosen_threshold, plots_dir):
    y_pred_proba = model.predict(X)
    y_pred = (y_pred_proba > chosen_threshold).astype(int)

    target_names = ['No Rechazo (0)', 'Rechazo (1)']

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix (Threshold={chosen_threshold:.2f})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    cm_plot_path = os.path.join(plots_dir, 'confusion_matrix.png')
    plt.savefig(cm_plot_path)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_plot_path}")

    # Classification Report Table
    report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    # Prepare DataFrame for table, matching example image structure for 'accuracy' row
    accuracy_value = report_dict.pop('accuracy') # Remove accuracy before creating DataFrame from dict of dicts
    report_df = pd.DataFrame(report_dict).transpose()
    
    # Add accuracy row formatted as in the example image
    if 'accuracy' not in report_df.index: # Ensure accuracy row is added if not present
        # Calculate total support for the accuracy row
        total_support = report_df.loc['macro avg', 'support'] if 'macro avg' in report_df.index else np.sum(report_df['support'][:-2]) # Sum of class supports

        accuracy_row = pd.DataFrame({
            'precision': [np.nan], 
            'recall': [np.nan], 
            'f1-score': [accuracy_value], 
            'support': [total_support]
        }, index=['accuracy'])
        report_df = pd.concat([report_df.iloc[:-2], accuracy_row, report_df.iloc[-2:]])


    report_df = report_df.round(3) # Round numbers for display

    fig, ax = plt.subplots(figsize=(8, (len(report_df) + 1) * 0.5)) # Adjust height based on number of rows
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=report_df.values,
                     colLabels=report_df.columns,
                     rowLabels=report_df.index,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.2, 0.2, 0.2, 0.2]) # Adjust column widths if necessary
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    # Add title above the table
    fig.suptitle(f'Classification Report (Threshold={chosen_threshold:.2f})', fontsize=14)
    plt.subplots_adjust(top=0.9) # Adjust top to make space for suptitle

    report_table_path = os.path.join(plots_dir, 'classification_report_min_cost.png')
    plt.savefig(report_table_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved classification report table to: {report_table_path}")

# Evaluate with a chosen threshold (e.g., 0.6 from image hint)
chosen_eval_threshold = 0.6
generate_evaluation_plots(model, X_test, y_test, chosen_eval_threshold, PLOTS_DIR)

# 11. Save model architecture plot - REMOVED
# try:
#     plot_path = os.path.join(PLOTS_DIR, 'model_architecture.png')
#     plot_model(
# model,
# to_file=plot_path,
# show_shapes=True,
# show_layer_names=True
# )
#     print(f"Saved model architecture plot to: {plot_path}")
# except ImportError as e:
#     print(f"Warning: Could not save model architecture plot. {str(e)}")

# 12. Plot and save training history (KEPT)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()

history_plot_path = os.path.join(PLOTS_DIR, 'training_history.png')
plt.savefig(history_plot_path)
plt.close()
print(f"Saved training history plot to: {history_plot_path}")

print("\nAll steps complete. Plots have been saved to:")
# print(f"- Models directory: {MODELS_DIR}") # Removed
print(f"- Plots directory: {PLOTS_DIR}")
# print(f"- Results directory: {RESULTS_DIR}") # Removed

