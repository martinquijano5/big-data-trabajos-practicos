import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Tensorflow imports for Functional API
from tensorflow.keras.models import Model # Changed from Sequential
from tensorflow.keras.layers import Input, Dense # Added Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical, plot_model # For quality classification & model plotting
from sklearn.metrics import classification_report, confusion_matrix
import datetime


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, 'modelo_color_calidad', f'modelo_color_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
os.makedirs(parent_dir, exist_ok=True)

# Cargar el archivo Excel
white_wine_data_path = os.path.join(current_dir, 'winequality-white.csv')
red_wine_data_path = os.path.join(current_dir, 'winequality-red.csv')

white_wine_data = pd.read_csv(white_wine_data_path, sep=';')
red_wine_data = pd.read_csv(red_wine_data_path, sep=';')

white_wine_data['type'] = 1 # 1 for white
red_wine_data['type'] = 0 # 0 for red

combined_data = pd.concat([white_wine_data, red_wine_data], axis=0)
print(combined_data.head())
print(combined_data.tail())

# Define features (X) and target variables (y)
feature_columns = [col for col in combined_data.columns if col not in ['quality', 'type']]
X = combined_data[feature_columns]
y_color = combined_data['type']

# --- Quality preparation ---
# Map quality scores to start from 0 for classification
quality_encoder = LabelEncoder()
y_quality_encoded = quality_encoder.fit_transform(combined_data['quality'])
num_quality_classes = len(np.unique(y_quality_encoded))
# One-hot encode quality labels
y_quality_categorical = to_categorical(y_quality_encoded, num_classes=num_quality_classes)

# Split data into training and testing sets
X_train, X_test, y_color_train, y_color_test, y_quality_train, y_quality_test = train_test_split(
    X, y_color, y_quality_categorical, test_size=0.2, random_state=42, stratify=y_color # Stratify by color
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Build Functional API Model ---
input_shape = (X_train_scaled.shape[1],)
inputs = Input(shape=input_shape, name='input_layer')

# Shared hidden layers
shared_layer_1 = Dense(32, activation='relu')(inputs)
shared_layer_2 = Dense(16, activation='relu')(shared_layer_1)

# Branch 1: Color prediction (binary classification)
color_output = Dense(1, activation='sigmoid', name='color_output')(shared_layer_2)

# Branch 2: Quality prediction (multi-class classification)
quality_output = Dense(num_quality_classes, activation='softmax', name='quality_output')(shared_layer_2)

# Create the Model
model = Model(inputs=inputs, outputs=[color_output, quality_output])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss={'color_output': 'binary_crossentropy', 'quality_output': 'categorical_crossentropy'},
              metrics={'color_output': 'accuracy', 'quality_output': 'accuracy'},
              loss_weights={'color_output': 0.5, 'quality_output': 0.5}) # Optional: weight losses

model.summary() # Print model architecture

# --- Save Model Architecture Plot ---
model_plot_path = os.path.join(parent_dir, 'model_architecture.png')
try:
    plot_model(model, to_file=model_plot_path, show_shapes=True, show_layer_names=True, show_dtype=False, show_layer_activations=True)
    print(f"Model architecture plot saved to: {model_plot_path}")
except ImportError as e:
    print(f"Could not plot model. Error: {e}. Make sure pydot and graphviz are installed.")

# --- Prepare target data for training ---
y_train_dict = {'color_output': y_color_train, 'quality_output': y_quality_train}
y_test_dict = {'color_output': y_color_test, 'quality_output': y_quality_test}

# --- Training ---
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train_scaled, y_train_dict,
                    validation_split=0.2, # Use part of training data for validation
                    epochs=100,
                    batch_size=32,
                    callbacks=[early_stopping],
                    verbose=1)

# --- Evaluation ---
loss, color_loss, quality_loss, color_acc, quality_acc = model.evaluate(X_test_scaled, y_test_dict, verbose=0)

print(f"\nTest Evaluation:")
print(f"  Overall Loss: {loss:.4f}")
print(f"  Color Loss: {color_loss:.4f}, Color Accuracy: {color_acc:.4f}")
print(f"  Quality Loss: {quality_loss:.4f}, Quality Accuracy: {quality_acc:.4f}")

# --- Predictions and Detailed Metrics ---
y_pred = model.predict(X_test_scaled)
y_color_pred_prob = y_pred[0]
y_quality_pred_prob = y_pred[1]

# Convert probabilities to class labels
y_color_pred = (y_color_pred_prob > 0.5).astype(int).flatten()
y_quality_pred = np.argmax(y_quality_pred_prob, axis=1)
y_quality_test_labels = np.argmax(y_quality_test, axis=1) # Convert one-hot back to labels for metrics

print("\nColor Prediction Metrics:")
print(classification_report(y_color_test, y_color_pred, target_names=['Red (0)', 'White (1)']))
print("Color Confusion Matrix:")
print(confusion_matrix(y_color_test, y_color_pred))


print("\nQuality Prediction Metrics:")
# Map encoded labels back to original quality scores for report
quality_target_names = [f'Quality {q}' for q in quality_encoder.inverse_transform(np.arange(num_quality_classes))]
# Explicitly define the labels corresponding to the target names
report_labels = np.arange(num_quality_classes)
# Add the 'labels' parameter to classification_report
print(classification_report(y_quality_test_labels, y_quality_pred, target_names=quality_target_names, labels=report_labels))
print("Quality Confusion Matrix:")
# Add the 'labels' parameter to confusion_matrix for consistency
print(confusion_matrix(y_quality_test_labels, y_quality_pred, labels=report_labels))


# --- Plotting Training History ---
# Plot accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['color_output_accuracy'], label='Train Color Acc')
plt.plot(history.history['val_color_output_accuracy'], label='Val Color Acc')
plt.plot(history.history['quality_output_accuracy'], label='Train Quality Acc')
plt.plot(history.history['val_quality_output_accuracy'], label='Val Quality Acc')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Overall Loss')
plt.plot(history.history['val_loss'], label='Val Overall Loss')
plt.plot(history.history['color_output_loss'], label='Train Color Loss')
plt.plot(history.history['val_color_output_loss'], label='Val Color Loss')
plt.plot(history.history['quality_output_loss'], label='Train Quality Loss')
plt.plot(history.history['val_quality_output_loss'], label='Val Quality Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
history_plot_path = os.path.join(parent_dir, 'training_history.png')
plt.savefig(history_plot_path)
print(f"\nTraining history plot saved to: {history_plot_path}")
plt.show()

# Save the model
model_path = os.path.join(parent_dir, 'color_quality_model.h5')
model.save(model_path)
print(f"Model saved to: {model_path}")

# ... after X_train_scaled, y_color_train, y_quality_train are defined
print(f"NaNs in X_train_scaled: {np.isnan(X_train_scaled).sum()}")
if isinstance(y_color_train, pd.Series):
    print(f"NaNs in y_color_train: {y_color_train.isnull().sum()}")
else:
    print(f"NaNs in y_color_train: {np.isnan(y_color_train).sum()}")
print(f"NaNs in y_quality_train: {np.isnan(y_quality_train).sum()}")

print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"y_color_train shape: {y_color_train.shape}")
print(f"y_quality_train shape: {y_quality_train.shape}")


plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')