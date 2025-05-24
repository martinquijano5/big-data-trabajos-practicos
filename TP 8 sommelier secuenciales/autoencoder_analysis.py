import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# --- Output directory setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
parent_dir = os.path.join(current_dir, 'autoencoder_analysis')
os.makedirs(parent_dir, exist_ok=True)

# --- Data loading and scaling ---
white_wine_data_path = os.path.join(current_dir, 'winequality-white.csv')
red_wine_data_path = os.path.join(current_dir, 'winequality-red.csv')

white_wine_data = pd.read_csv(white_wine_data_path, sep=';')
red_wine_data = pd.read_csv(red_wine_data_path, sep=';')
white_wine_data['type'] = 1
red_wine_data['type'] = 0
combined_data = pd.concat([white_wine_data, red_wine_data], axis=0)

feature_columns = [col for col in combined_data.columns if col not in ['type', 'quality']]
X_original_features = combined_data[feature_columns]
y_labels = combined_data[['type', 'quality']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_original_features)

# --- Autoencoder definition and training ---
def plot_autoencoder_loss(history, parent_dir):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Reconstruction Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plot_path = os.path.join(parent_dir, 'autoencoder_loss_plot.png')
    plt.savefig(plot_path)
    print(f"Autoencoder loss plot saved to {plot_path}")
    plt.close()

def build_and_train_autoencoder(data_train, data_val, original_dim, encoding_dim, epochs=50, batch_size=32):
    input_features = Input(shape=(original_dim,), name='encoder_input')
    encoded = Dense(encoding_dim, activation='relu', name='encoded_layer')(input_features)
    encoder = Model(input_features, encoded, name='encoder')

    encoded_input = Input(shape=(encoding_dim,), name='decoder_input')
    decoded = Dense(original_dim, activation=None, name='decoder_output')(encoded_input)
    decoder = Model(encoded_input, decoded, name='decoder')

    autoencoder_input = Input(shape=(original_dim,), name='autoencoder_input')
    encoded_output = encoder(autoencoder_input)
    decoded_output = decoder(encoded_output)
    autoencoder = Model(autoencoder_input, decoded_output, name='autoencoder')

    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    print("Training autoencoder...")
    autoencoder.summary()
    history = autoencoder.fit(data_train, data_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              shuffle=True,
                              validation_data=(data_val, data_val),
                              verbose=1,
                              callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
    print("Autoencoder training finished.")
    return encoder, autoencoder, history

def plot_original_vs_reconstructed(X_original_scaled, autoencoder_model, feature_names, parent_dir):
    """
    Plots scatter plots of original scaled features vs. their reconstructions.
    """
    X_reconstructed = autoencoder_model.predict(X_original_scaled)
    
    num_features = X_original_scaled.shape[1]
    # Determine a reasonable number of plots to avoid clutter, e.g., max 6-8
    features_to_plot_indices = np.random.choice(num_features, size=min(num_features, 8), replace=False)

    cols = 2 # Number of columns in subplot grid
    rows = (len(features_to_plot_indices) + cols - 1) // cols # Calculate rows needed

    plt.figure(figsize=(7 * cols, 6 * rows)) 
    for i, feature_idx in enumerate(features_to_plot_indices):
        plt.subplot(rows, cols, i + 1)
        plt.scatter(X_original_scaled[:, feature_idx], X_reconstructed[:, feature_idx], alpha=0.3, s=10, label='Actual vs. Reconstructed')
        # Add a y=x line for reference
        lim_min = np.min([X_original_scaled[:, feature_idx].min(), X_reconstructed[:, feature_idx].min()])
        lim_max = np.max([X_original_scaled[:, feature_idx].max(), X_reconstructed[:, feature_idx].max()])
        plt.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', alpha=0.75, zorder=0, label='Ideal (y=x)')
        plt.xlabel(f"Original Scaled: {feature_names[feature_idx]}")
        plt.ylabel(f"Reconstructed: {feature_names[feature_idx]}")
        plt.title(f"Reconstruction of {feature_names[feature_idx]}")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(parent_dir, 'original_vs_reconstructed_scatter.png')
    plt.savefig(plot_path)
    print(f"Original vs. Reconstructed scatter plots saved to {plot_path}")
    plt.close()

def analyze_variance_preservation(X_original_scaled, autoencoder_model, original_feature_names, parent_dir):
    """
    Analyzes and visualizes variance preservation by comparing original scaled features
    to their reconstructions from the autoencoder. Saves results to CSV and table image.
    """
    print("\n--- Analyzing Variance Preservation ---")
    X_reconstructed = autoencoder_model.predict(X_original_scaled)

    # Total variance
    total_variance_original = np.sum(np.var(X_original_scaled, axis=0))
    total_variance_reconstructed = np.sum(np.var(X_reconstructed, axis=0))

    print(f"Total variance of original scaled data: {total_variance_original:.4f}")
    print(f"Total variance of reconstructed data: {total_variance_reconstructed:.4f}")
    
    percentage_retained_str = "N/A (Original variance is too low)"
    if total_variance_original > 1e-9: # Avoid division by zero or tiny numbers
        percentage_retained = (total_variance_reconstructed / total_variance_original) * 100
        percentage_retained_str = f"{percentage_retained:.2f}%"
        print(f"Percentage of total variance retained in reconstruction: {percentage_retained_str}")
    else:
        print("Original total variance is very close to zero, percentage retained cannot be reliably calculated.")

    # Per-feature variance
    variances_original = np.var(X_original_scaled, axis=0)
    variances_reconstructed = np.var(X_reconstructed, axis=0)

    df_variances = pd.DataFrame({
        'Feature': original_feature_names,
        'Original Scaled Variance': variances_original,
        'Reconstructed Variance': variances_reconstructed
    })
    df_variances['Variance Retained (%)'] = (df_variances['Reconstructed Variance'] / df_variances['Original Scaled Variance']) * 100
    df_variances['Variance Retained (%)'] = df_variances['Variance Retained (%)'].fillna(0).round(2) # Handle potential division by zero if original variance is 0

    # Save total variance summary to text file
    summary_text_path = os.path.join(parent_dir, 'total_variance_summary.txt')
    with open(summary_text_path, 'w') as f:
        f.write("Total Variance Summary:\n")
        f.write("-------------------------\n")
        f.write(f"Total variance of original scaled data: {total_variance_original:.4f}\n")
        f.write(f"Total variance of reconstructed data: {total_variance_reconstructed:.4f}\n")
        f.write(f"Percentage of total variance retained in reconstruction: {percentage_retained_str}\n")
    print(f"Total variance summary saved to {summary_text_path}")

    # Plotting per-feature variances (bar chart - already exists, ensure it's still useful)
    plt.figure(figsize=(max(12, len(original_feature_names) * 0.8), 7)) 
    df_variances_plot = df_variances[['Feature', 'Original Scaled Variance', 'Reconstructed Variance']].copy()
    df_variances_plot.set_index('Feature').plot(kind='bar', ax=plt.gca())
    plt.title('Variance Comparison: Original Scaled vs. Reconstructed Features')
    plt.ylabel('Variance')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plot_path = os.path.join(parent_dir, 'feature_variance_comparison_bar_chart.png') # Renamed for clarity
    plt.savefig(plot_path)
    print(f"Per-feature variance comparison bar chart saved to {plot_path}")
    plt.close()

    # Create and save a table image of df_variances
    fig, ax = plt.subplots(figsize=(max(10, len(original_feature_names)*0.5), max(4, df_variances.shape[0] * 0.3 + 1))) # Adjust size
    ax.axis('tight')
    ax.axis('off')
    # Format float columns for better display in the table image
    df_table_display = df_variances.copy()
    for col in ['Original Scaled Variance', 'Reconstructed Variance', 'Variance Retained (%)']:
        df_table_display[col] = df_table_display[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
        if col == 'Variance Retained (%)':
             df_table_display[col] = df_table_display[col].apply(lambda x: f"{x}%" if pd.notna(x) and x != 'nan' else "N/A")


    table_data = ax.table(cellText=df_table_display.values,
                          colLabels=df_table_display.columns,
                          cellLoc = 'center',
                          loc='center')
    table_data.auto_set_font_size(False)
    table_data.set_fontsize(8)
    table_data.scale(1.2, 1.2) # Adjust scale for better fit
    plt.title('Feature Variance Details: Original vs. Reconstructed', y=1.08) # Add title
    table_image_path = os.path.join(parent_dir, 'feature_variance_table.png')
    plt.savefig(table_image_path, bbox_inches='tight', dpi=200)
    print(f"Feature variance table image saved to {table_image_path}")
    plt.close()

def plot_encoded_vs_target_correlations(X_encoded_df_with_targets, encoded_feature_names, parent_dir):
    """
    Calculates and plots the correlation of encoded features with target variables (type and quality).
    """
    print("\n--- Analyzing Correlation of Encoded Features with Target Variables ---")

    if 'type' not in X_encoded_df_with_targets.columns or 'quality' not in X_encoded_df_with_targets.columns:
        print("Error: 'type' or 'quality' column not found in the DataFrame for correlation analysis.")
        return

    # Correlation with 'type'
    correlations_type = X_encoded_df_with_targets[encoded_feature_names].corrwith(X_encoded_df_with_targets['type'])
    print("\nCorrelation of Encoded Features with 'type':")
    print(correlations_type)

    plt.figure(figsize=(max(8, len(encoded_feature_names) * 0.6), 6))
    correlations_type.sort_values().plot(kind='barh') # Horizontal bar plot for better readability
    plt.title('Correlation of Encoded Features with Wine Type')
    plt.xlabel('Pearson Correlation Coefficient')
    plt.ylabel('Encoded Feature')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path_type = os.path.join(parent_dir, 'correlation_encoded_vs_type.png')
    plt.savefig(plot_path_type)
    print(f"Correlation plot (encoded vs type) saved to {plot_path_type}")
    plt.close()

    # Correlation with 'quality'
    correlations_quality = X_encoded_df_with_targets[encoded_feature_names].corrwith(X_encoded_df_with_targets['quality'])
    print("\nCorrelation of Encoded Features with 'quality':")
    print(correlations_quality)

    plt.figure(figsize=(max(8, len(encoded_feature_names) * 0.6), 6))
    correlations_quality.sort_values().plot(kind='barh') # Horizontal bar plot
    plt.title('Correlation of Encoded Features with Wine Quality')
    plt.xlabel('Pearson Correlation Coefficient')
    plt.ylabel('Encoded Feature')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path_quality = os.path.join(parent_dir, 'correlation_encoded_vs_quality.png')
    plt.savefig(plot_path_quality)
    print(f"Correlation plot (encoded vs quality) saved to {plot_path_quality}")
    plt.close()

# Split for AE validation
X_train_for_ae, X_val_for_ae = train_test_split(X_scaled, test_size=0.2, random_state=42)
original_dim = X_scaled.shape[1]
encoding_dim = int(original_dim / 2)
encoder_model, autoencoder_model, ae_history = build_and_train_autoencoder(
    X_train_for_ae, X_val_for_ae, original_dim, encoding_dim, epochs=50
)
plot_autoencoder_loss(ae_history, parent_dir)

# --- Encode the full dataset ---
X_encoded = encoder_model.predict(X_scaled)
encoded_feature_names = [f'encoded_dim_{i}' for i in range(encoding_dim)]
X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_feature_names)
X_encoded_df['type'] = y_labels['type'].values
X_encoded_df['quality'] = y_labels['quality'].values

# --- Visualizations ---
# 2D scatter of first two encoded dims
if encoding_dim >= 2:
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_encoded_df['encoded_dim_0'], X_encoded_df['encoded_dim_1'], c=X_encoded_df['type'], cmap='coolwarm', alpha=0.6)
    plt.xlabel('encoded_dim_0')
    plt.ylabel('encoded_dim_1')
    plt.title('2D Scatter of Encoded Features (colored by type)')
    plt.colorbar(scatter, label='type')
    plt.savefig(os.path.join(parent_dir, 'scatter_encoded_2d_type.png'))
    plt.close()
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_encoded_df['encoded_dim_0'], X_encoded_df['encoded_dim_1'], c=X_encoded_df['quality'], cmap='viridis', alpha=0.6)
    plt.xlabel('encoded_dim_0')
    plt.ylabel('encoded_dim_1')
    plt.title('2D Scatter of Encoded Features (colored by quality)')
    plt.colorbar(scatter, label='quality')
    plt.savefig(os.path.join(parent_dir, 'scatter_encoded_2d_quality.png'))
    plt.close()

# New: Plot Original vs. Reconstructed
plot_original_vs_reconstructed(X_scaled, autoencoder_model, feature_columns, parent_dir)

# New: Analyze Variance Preservation
analyze_variance_preservation(X_scaled, autoencoder_model, feature_columns, parent_dir)

# --- Correlation Matrix ---
corr = X_encoded_df[encoded_feature_names].corr()
plt.figure(figsize=(min(12, 1+encoding_dim), min(10, 1+encoding_dim)))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix of Encoded Features')
plt.tight_layout()
plt.savefig(os.path.join(parent_dir, 'encoded_feature_correlation_heatmap.png'))
plt.close()
print(f"Correlation heatmap saved to {os.path.join(parent_dir, 'encoded_feature_correlation_heatmap.png')}")

# New: Analyze Correlation with Target Variables
# X_encoded_df already contains 'type' and 'quality' columns at this point
plot_encoded_vs_target_correlations(X_encoded_df, encoded_feature_names, parent_dir)

print("Autoencoder analysis complete. All results saved to:", parent_dir) 