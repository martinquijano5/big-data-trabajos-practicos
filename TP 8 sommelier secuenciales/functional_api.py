import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Tensorflow imports for Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import tensorflow as tf

# Function to plot autoencoder loss
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

# Function to build and train the autoencoder
def build_and_train_autoencoder(data_train, data_val, original_dim, encoding_dim, epochs=50, batch_size=32):
    # --- Encoder ---
    input_features = Input(shape=(original_dim,), name='encoder_input')
    # Single dense layer for encoder
    encoded = Dense(encoding_dim, activation='relu', name='encoded_layer')(input_features)
    encoder = Model(input_features, encoded, name='encoder')

    # --- Decoder ---
    encoded_input = Input(shape=(encoding_dim,), name='decoder_input')
    # Single dense layer for decoder
    decoded = Dense(original_dim, activation='sigmoid', name='decoder_output')(encoded_input) # Sigmoid for reconstruction if data is scaled 0-1, or linear/relu otherwise
                                                                                         # Assuming X_scaled is from StandardScaler, sigmoid might not be best.
                                                                                         # Let's use linear for reconstruction of standardized data, or relu.
                                                                                         # Or, more commonly, no activation or 'linear' for the final reconstruction.
    # For standardized data (from StandardScaler), a linear activation is often better.
    decoded = Dense(original_dim, activation=None, name='decoder_output')(encoded_input)


    decoder = Model(encoded_input, decoded, name='decoder')

    # --- Autoencoder (Encoder + Decoder) ---
    autoencoder_input = Input(shape=(original_dim,), name='autoencoder_input')
    encoded_output = encoder(autoencoder_input)
    decoded_output = decoder(encoded_output)
    autoencoder = Model(autoencoder_input, decoded_output, name='autoencoder')

    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse') # Mean Squared Error for reconstruction
    
    print("Training autoencoder...")
    autoencoder.summary() # Print autoencoder structure

    history = autoencoder.fit(data_train, data_train, # Autoencoder learns to reconstruct its input
                              epochs=epochs,
                              batch_size=batch_size,
                              shuffle=True,
                              validation_data=(data_val, data_val),
                              verbose=1,
                              callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
    print("Autoencoder training finished.")
    return encoder, autoencoder, history

# Function to plot loss and accuracy from history
def plot_loss_and_accuracy(history, parent_dir):
    history_dict = history.history
    epochs_range = range(1, len(history_dict['loss']) + 1)

    plt.figure(figsize=(18, 12))

    # Plot Overall Loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs_range, history_dict['loss'], 'bo-', label='Training Loss')
    plt.plot(epochs_range, history_dict['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Overall Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Type Output Loss
    plt.subplot(2, 3, 2)
    plt.plot(epochs_range, history_dict['type_output_loss'], 'bo-', label='Training Type Loss')
    plt.plot(epochs_range, history_dict['val_type_output_loss'], 'ro-', label='Validation Type Loss')
    plt.title('Type Output: Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Quality Output Loss
    plt.subplot(2, 3, 3)
    plt.plot(epochs_range, history_dict['quality_output_loss'], 'bo-', label='Training Quality Loss')
    plt.plot(epochs_range, history_dict['val_quality_output_loss'], 'ro-', label='Validation Quality Loss')
    plt.title('Quality Output: Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Type Output Accuracy
    plt.subplot(2, 3, 4)
    plt.plot(epochs_range, history_dict.get('type_output_type_accuracy', history_dict.get('type_accuracy')), 'bo-', label='Training Type Accuracy')
    plt.plot(epochs_range, history_dict.get('val_type_output_type_accuracy', history_dict.get('val_type_accuracy')), 'ro-', label='Validation Type Accuracy')
    plt.title('Type Output: Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Quality Output Accuracy
    plt.subplot(2, 3, 5)
    plt.plot(epochs_range, history_dict.get('quality_output_quality_accuracy', history_dict.get('quality_accuracy')), 'bo-', label='Training Quality Accuracy')
    plt.plot(epochs_range, history_dict.get('val_quality_output_quality_accuracy', history_dict.get('val_quality_accuracy')), 'ro-', label='Validation Quality Accuracy')
    plt.title('Quality Output: Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(parent_dir, 'training_loss_accuracy_plots.png')
    plt.savefig(plot_path)
    print(f"Loss and accuracy plots saved to {plot_path}")
    plt.close()

# Function to save classification reports as table images
def save_classification_reports(model, X_test, y_test_type, y_test_quality_df, quality_target_names, parent_dir):
    y_pred_type_prob, y_pred_quality_prob = model.predict(X_test)

    # --- Type Classification Report Image ---
    y_pred_type = (y_pred_type_prob > 0.5).astype(int)
    type_target_names = ['Red (0)', 'White (1)']
    report_type_dict = classification_report(y_test_type, y_pred_type, target_names=type_target_names, output_dict=True, zero_division=0)
    
    accuracy_score_type = report_type_dict.pop('accuracy', None)
    df_metrics_type = pd.DataFrame(report_type_dict).transpose()

    df_ordered_report_type_list = []
    # Add class rows in specified order
    for target_name in type_target_names:
        if target_name in df_metrics_type.index:
            df_ordered_report_type_list.append(df_metrics_type.loc[[target_name]])
    
    # Add accuracy row
    if accuracy_score_type is not None:
        total_support_type = y_test_type.shape[0]
        accuracy_df_row_type = pd.DataFrame(
            {'precision': np.nan, 'recall': np.nan, 'f1-score': accuracy_score_type, 'support': total_support_type},
            index=['accuracy']
        )
        df_ordered_report_type_list.append(accuracy_df_row_type)

    # Add average rows
    for avg_key in ['macro avg', 'weighted avg']:
        if avg_key in df_metrics_type.index:
            df_ordered_report_type_list.append(df_metrics_type.loc[[avg_key]])
    
    if df_ordered_report_type_list:
        df_report_type = pd.concat(df_ordered_report_type_list)
    else: # Fallback if all lists are empty (should not happen)
        df_report_type = pd.DataFrame()


    df_report_type = df_report_type.round(3)

    if not df_report_type.empty:
        fig_type_height = max(3, len(df_report_type.index) * 0.4 + 1) # Dynamic height based on num rows
        fig_type, ax_type = plt.subplots(figsize=(8, fig_type_height)) 
        ax_type.axis('tight')
        ax_type.axis('off')
        table_type = ax_type.table(cellText=df_report_type.values,
                                   colLabels=df_report_type.columns,
                                   rowLabels=df_report_type.index,
                                   cellLoc='center',
                                   loc='center',
                                   colWidths=[0.2, 0.2, 0.2, 0.2]) # Adjust colWidths if needed
        table_type.auto_set_font_size(False)
        table_type.set_fontsize(10)
        table_type.scale(1.1, 1.1) 
        ax_type.set_title("Classification Report for Wine Type", y=0.95, fontsize=14, pad=20)
        report_type_image_path = os.path.join(parent_dir, 'classification_report_type.png')
        plt.savefig(report_type_image_path, bbox_inches='tight', dpi=200)
        plt.close(fig_type) 
        print(f"Type classification report image saved to {report_type_image_path}")
    else:
        print("Could not generate type classification report table: DataFrame is empty.")


    # --- Quality Classification Report Image ---
    y_pred_quality_labels = np.argmax(y_pred_quality_prob, axis=1)
    y_test_quality_labels = np.argmax(y_test_quality_df.values, axis=1)
    quality_labels_range = list(range(len(quality_target_names)))

    report_quality_dict = classification_report(y_test_quality_labels,
                                               y_pred_quality_labels,
                                               target_names=quality_target_names,
                                               labels=quality_labels_range,
                                               zero_division=0,
                                               output_dict=True)
    
    accuracy_score_quality = report_quality_dict.pop('accuracy', None)
    df_metrics_quality = pd.DataFrame(report_quality_dict).transpose()

    df_ordered_report_quality_list = []
    # Add class rows in specified order (quality_target_names)
    for target_name in quality_target_names:
        if target_name in df_metrics_quality.index:
            df_ordered_report_quality_list.append(df_metrics_quality.loc[[target_name]])
        else: # Add a row of NaNs if class was not in report (e.g. 0 support and not predicted)
            nan_row = pd.DataFrame(
                {'precision': np.nan, 'recall': np.nan, 'f1-score': np.nan, 'support': 0}, # Support is 0
                index=[target_name]
            )
            df_ordered_report_quality_list.append(nan_row)

    # Add accuracy row
    if accuracy_score_quality is not None:
        total_support_quality = y_test_quality_labels.shape[0]
        accuracy_df_row_quality = pd.DataFrame(
            {'precision': np.nan, 'recall': np.nan, 'f1-score': accuracy_score_quality, 'support': total_support_quality},
            index=['accuracy']
        )
        df_ordered_report_quality_list.append(accuracy_df_row_quality)

    # Add average rows
    for avg_key in ['macro avg', 'weighted avg']:
        if avg_key in df_metrics_quality.index:
            df_ordered_report_quality_list.append(df_metrics_quality.loc[[avg_key]])
            
    if df_ordered_report_quality_list:
        df_report_quality = pd.concat(df_ordered_report_quality_list)
    else: # Fallback
        df_report_quality = pd.DataFrame()

    df_report_quality = df_report_quality.round(3)

    if not df_report_quality.empty:
        fig_quality_height = max(5, len(df_report_quality.index) * 0.4 + 1) # Dynamic height
        fig_quality, ax_quality = plt.subplots(figsize=(10, fig_quality_height))
        ax_quality.axis('tight')
        ax_quality.axis('off')
        table_quality = ax_quality.table(cellText=df_report_quality.values,
                                         colLabels=df_report_quality.columns,
                                         rowLabels=df_report_quality.index,
                                         cellLoc='center',
                                         loc='center',
                                         colWidths=[0.2, 0.2, 0.2, 0.2])
        table_quality.auto_set_font_size(False)
        table_quality.set_fontsize(10)
        table_quality.scale(1.1, 1.1)
        ax_quality.set_title("Classification Report for Wine Quality", y=0.95, fontsize=14, pad=20) 
        report_quality_image_path = os.path.join(parent_dir, 'classification_report_quality.png')
        plt.savefig(report_quality_image_path, bbox_inches='tight', dpi=200)
        plt.close(fig_quality)
        print(f"Quality classification report image saved to {report_quality_image_path}")
    else:
        print("Could not generate quality classification report table: DataFrame is empty.")

# Function to plot and save confusion matrices
def plot_confusion_matrices(model, X_test, y_test_type, y_test_quality_df, quality_target_names, parent_dir):
    y_pred_type_prob, y_pred_quality_prob = model.predict(X_test)

    # Type confusion matrix
    y_pred_type = (y_pred_type_prob > 0.5).astype(int)
    cm_type = confusion_matrix(y_test_type, y_pred_type)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_type, annot=True, fmt='d', cmap='Blues', xticklabels=['Red (0)', 'White (1)'], yticklabels=['Red (0)', 'White (1)'])
    plt.title('Confusion Matrix for Wine Type')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    cm_type_path = os.path.join(parent_dir, 'confusion_matrix_type.png')
    plt.savefig(cm_type_path)
    print(f"Type confusion matrix saved to {cm_type_path}")
    plt.close()

    # Quality confusion matrix
    y_pred_quality_labels = np.argmax(y_pred_quality_prob, axis=1)
    y_test_quality_labels = np.argmax(y_test_quality_df.values, axis=1)
    
    # Define the full range of possible labels for quality
    quality_labels_range = list(range(len(quality_target_names)))

    cm_quality = confusion_matrix(y_test_quality_labels, y_pred_quality_labels, labels=quality_labels_range)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_quality, annot=True, fmt='d', cmap='Greens', xticklabels=quality_target_names, yticklabels=quality_target_names)
    plt.title('Confusion Matrix for Wine Quality')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    cm_quality_path = os.path.join(parent_dir, 'confusion_matrix_quality.png')
    plt.savefig(cm_quality_path)
    print(f"Quality confusion matrix saved to {cm_quality_path}")
    plt.close()

# Function to save quality classification report with tolerance
def save_quality_classification_report_with_tolerance(model, X_test, y_test_quality_df, quality_target_names, parent_dir):
    _, y_pred_quality_prob = model.predict(X_test) # Get only quality predictions

    y_pred_original_indices = np.argmax(y_pred_quality_prob, axis=1)
    y_true_indices = np.argmax(y_test_quality_df.values, axis=1)

    # Adjust predictions for tolerance: if prediction is within +/-1 of true, count as true
    y_pred_effective_indices_for_tolerance = y_pred_original_indices.copy()
    for i in range(len(y_true_indices)):
        if abs(y_pred_original_indices[i] - y_true_indices[i]) <= 1:
            y_pred_effective_indices_for_tolerance[i] = y_true_indices[i]
    
    quality_labels_range = list(range(len(quality_target_names)))

    report_quality_dict = classification_report(y_true_indices,
                                               y_pred_effective_indices_for_tolerance, # Use adjusted predictions
                                               target_names=quality_target_names,
                                               labels=quality_labels_range,
                                               zero_division=0,
                                               output_dict=True)
    
    accuracy_score_quality = report_quality_dict.pop('accuracy', None)
    df_metrics_quality = pd.DataFrame(report_quality_dict).transpose()

    df_ordered_report_quality_list = []
    for target_name in quality_target_names:
        if target_name in df_metrics_quality.index:
            df_ordered_report_quality_list.append(df_metrics_quality.loc[[target_name]])
        else:
            nan_row = pd.DataFrame(
                {'precision': np.nan, 'recall': np.nan, 'f1-score': np.nan, 'support': 0},
                index=[target_name]
            )
            df_ordered_report_quality_list.append(nan_row)

    if accuracy_score_quality is not None:
        total_support_quality = y_true_indices.shape[0]
        accuracy_df_row_quality = pd.DataFrame(
            {'precision': np.nan, 'recall': np.nan, 'f1-score': accuracy_score_quality, 'support': total_support_quality},
            index=['accuracy']
        )
        df_ordered_report_quality_list.append(accuracy_df_row_quality)

    for avg_key in ['macro avg', 'weighted avg']:
        if avg_key in df_metrics_quality.index:
            df_ordered_report_quality_list.append(df_metrics_quality.loc[[avg_key]])
            
    if df_ordered_report_quality_list:
        df_report_quality = pd.concat(df_ordered_report_quality_list)
    else:
        df_report_quality = pd.DataFrame()

    df_report_quality = df_report_quality.round(3)

    if not df_report_quality.empty:
        fig_quality_height = max(5, len(df_report_quality.index) * 0.4 + 1)
        fig_quality, ax_quality = plt.subplots(figsize=(10, fig_quality_height))
        ax_quality.axis('tight')
        ax_quality.axis('off')
        table_quality = ax_quality.table(cellText=df_report_quality.values,
                                         colLabels=df_report_quality.columns,
                                         rowLabels=df_report_quality.index,
                                         cellLoc='center',
                                         loc='center',
                                         colWidths=[0.2, 0.2, 0.2, 0.2])
        table_quality.auto_set_font_size(False)
        table_quality.set_fontsize(10)
        table_quality.scale(1.1, 1.1)
        ax_quality.set_title("Quality Classification Report (with +/-1 Tolerance)", y=0.95, fontsize=14, pad=20) 
        report_quality_image_path = os.path.join(parent_dir, 'classification_report_quality_with_tolerance.png')
        plt.savefig(report_quality_image_path, bbox_inches='tight', dpi=200)
        plt.close(fig_quality)
        print(f"Quality classification report with tolerance (image) saved to {report_quality_image_path}")
    else:
        print("Could not generate quality classification report table with tolerance: DataFrame is empty.")

# Function to plot and save quality confusion matrix with tolerance
def plot_quality_confusion_matrix_with_tolerance(model, X_test, y_test_quality_df, quality_target_names, parent_dir):
    _, y_pred_quality_prob = model.predict(X_test) # Get only quality predictions

    y_pred_original_indices = np.argmax(y_pred_quality_prob, axis=1)
    y_true_indices = np.argmax(y_test_quality_df.values, axis=1)

    # Adjust predictions for tolerance: if prediction is within +/-1 of true, count as true for CM
    y_pred_effective_indices_for_tolerance = y_pred_original_indices.copy()
    for i in range(len(y_true_indices)):
        if abs(y_pred_original_indices[i] - y_true_indices[i]) <= 1:
            y_pred_effective_indices_for_tolerance[i] = y_true_indices[i]

    quality_labels_range = list(range(len(quality_target_names)))

    cm_quality_tolerant = confusion_matrix(y_true_indices, 
                                           y_pred_effective_indices_for_tolerance, # Use adjusted predictions
                                           labels=quality_labels_range)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_quality_tolerant, annot=True, fmt='d', cmap='Purples', 
                xticklabels=quality_target_names, yticklabels=quality_target_names)
    plt.title('Confusion Matrix for Wine Quality (with +/-1 Tolerance)')
    plt.xlabel('Predicted Label (Adjusted for Tolerance)')
    plt.ylabel('True Label')
    cm_quality_tolerant_path = os.path.join(parent_dir, 'confusion_matrix_quality_with_tolerance.png')
    plt.savefig(cm_quality_tolerant_path)
    print(f"Quality confusion matrix with tolerance saved to {cm_quality_tolerant_path}")
    plt.close()

# Helper function to create PCA explained variance table
def create_pca_explained_variance_table(pca, parent_dir):
    explained_variance_ratios = pca.explained_variance_ratio_
    df_data = {
        'Principal Component': ['PC1', 'PC2', 'Total Explained (PC1+PC2)'],
        'Variance Explained Ratio': [
            explained_variance_ratios[0],
            explained_variance_ratios[1],
            np.sum(explained_variance_ratios[:2])
        ]
    }
    df_explained_variance = pd.DataFrame(df_data)
    df_explained_variance['Variance Explained Ratio'] = df_explained_variance['Variance Explained Ratio'].round(4)

    fig, ax = plt.subplots(figsize=(6, 2)) # Adjust size as needed
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df_explained_variance.values,
                     colLabels=df_explained_variance.columns,
                     loc='center',
                     cellLoc='center',
                     colWidths=[0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.1)
    ax.set_title("PCA Explained Variance Ratio", fontsize=14, y=0.7) # Adjust title position
    plt.savefig(os.path.join(parent_dir, 'pca_explained_variance_table.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"PCA explained variance table saved to {os.path.join(parent_dir, 'pca_explained_variance_table.png')}")

# Helper function to create PCA component loadings table
def create_pca_component_loadings_table(pca, feature_names, parent_dir):
    loadings = pca.components_[:2, :].T # PC1 and PC2 loadings for each feature
    df_loadings = pd.DataFrame(loadings, columns=['PC1 Loading', 'PC2 Loading'], index=feature_names)
    df_loadings = df_loadings.round(4)

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.3 + 1))) # Dynamic height
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df_loadings.values,
                     colLabels=df_loadings.columns,
                     rowLabels=df_loadings.index,
                     loc='center',
                     cellLoc='center',
                     colWidths=[0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.1)
    ax.set_title(f"PCA Component Loadings ({len(feature_names)} Features)", fontsize=14, y=0.95, pad=20)
    plt.savefig(os.path.join(parent_dir, 'pca_component_loadings_table.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"PCA component loadings table saved to {os.path.join(parent_dir, 'pca_component_loadings_table.png')}")

# Helper function to define and train a model on 2D PCA features
def train_pca_model(X_train_pca, y_train_type, y_train_quality_one_hot, num_quality_classes):
    input_pca = Input(shape=(2,), name='pca_input_layer')
    hidden_pca = Dense(16, activation='relu', name='pca_hidden_1')(input_pca)
    # hidden_pca_2 = Dense(8, activation='relu', name='pca_hidden_2')(hidden_pca) # Optional deeper
    
    type_output_pca = Dense(1, activation='sigmoid', name='pca_type_output')(hidden_pca)
    quality_output_pca = Dense(num_quality_classes, activation='softmax', name='pca_quality_output')(hidden_pca)
    
    model_pca = Model(inputs=input_pca, outputs=[type_output_pca, quality_output_pca])
    
    model_pca.compile(optimizer=Adam(learning_rate=0.001),
                      loss={'pca_type_output': 'binary_crossentropy', 'pca_quality_output': 'categorical_crossentropy'},
                      metrics={'pca_type_output': 'accuracy', 'pca_quality_output': 'accuracy'})
    
    print("Training model on PCA features...")
    model_pca.fit(X_train_pca,
                  {'pca_type_output': y_train_type, 'pca_quality_output': y_train_quality_one_hot},
                  epochs=50, # Adjust as needed
                  batch_size=32,
                  verbose=1,
                  callbacks=[EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)])
    print("PCA model training finished.")
    return model_pca

# Helper function to plot decision boundaries
def plot_decision_boundary(ax, model_pca, X_test_pca, y_test_labels_for_scatter,
                           model_output_name, plot_title, class_names, parent_dir, filename,
                           is_quality_binned=False, quality_bin_mapping_func=None, min_original_quality_score_for_binning=None):
    h = .02  
    x_min, x_max = X_test_pca[:, 0].min() - 0.5, X_test_pca[:, 0].max() + 0.5
    y_min, y_max = X_test_pca[:, 1].min() - 0.5, X_test_pca[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    full_predictions = model_pca.predict(mesh_points, verbose=1)
    if model_output_name == 'pca_type_output':
        Z_probs = full_predictions[0] 
        Z = (Z_probs > 0.5).astype(int).reshape(xx.shape)
        num_contour_classes = 2 
    else: # pca_quality_output
        Z_probs = full_predictions[1] 
        Z_predicted_indices = np.argmax(Z_probs, axis=1) 
        if is_quality_binned and quality_bin_mapping_func:
            Z_original_scores = Z_predicted_indices + min_original_quality_score_for_binning
            Z = np.array([quality_bin_mapping_func(score) for score in Z_original_scores]).reshape(xx.shape)
            num_contour_classes = len(class_names) 
        else:
            Z = Z_predicted_indices.reshape(xx.shape)
            num_contour_classes = len(class_names) 
            
    cmap_light = plt.cm.get_cmap('viridis', num_contour_classes if num_contour_classes > 1 else 2)
    contour = ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6) # Store the contourf output
    
    scatter_cmap_classes = len(np.unique(y_test_labels_for_scatter))
    cmap_bold = plt.cm.get_cmap('plasma', scatter_cmap_classes if scatter_cmap_classes > 1 else 2) 
    
    scatter = ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test_labels_for_scatter,
                       cmap=cmap_bold, edgecolor='k', s=20, alpha=0.8)
    
    ax.set_title(plot_title)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    if class_names:
        # Legend for scatter points (True Labels)
        num_scatter_legend_items = len(np.unique(y_test_labels_for_scatter))
        # If class_names is longer than unique scatter labels (e.g. binned quality vs full quality names)
        # use the length of actual unique items for scatter legend colors
        effective_scatter_class_names = class_names
        if is_quality_binned: # For binned quality, class_names is already binned_quality_names (Low, Medium, High)
             num_scatter_legend_items = len(class_names) # Use class_names which holds binned_quality_names
             effective_scatter_class_names = class_names # Use class_names which holds binned_quality_names
        elif model_output_name == 'pca_type_output':
             num_scatter_legend_items = 2
             effective_scatter_class_names = ['Red (0)', 'White (1)']
        else: # Normal quality plot
            # Ensure we use the actual number of unique quality scores present in y_test_labels_for_scatter
            # y_test_labels_for_scatter are indices 0-6 for normal quality
            # class_names are ['quality_3', ..., 'quality_9']
            # We need to map these correctly for legend
            num_scatter_legend_items = len(class_names) # This should be 7 for normal quality
            effective_scatter_class_names = class_names


        legend_cmap_for_scatter_handles = plt.cm.get_cmap('plasma', num_scatter_legend_items if num_scatter_legend_items > 1 else 2)
        
        scatter_legend_handles = []
        # Ensure we only create legend items for classes actually present in the scatter data if y_test_labels_for_scatter are indices
        # For binned, y_test_labels_for_scatter are already 0,1,2
        # For type, y_test_labels_for_scatter are 0,1
        # For normal quality, y_test_labels_for_scatter are 0-6 (indices)

        unique_scatter_labels = np.unique(y_test_labels_for_scatter)

        for i, label_val in enumerate(unique_scatter_labels):
            # Determine the correct class name for the legend
            current_class_name = ""
            if is_quality_binned:
                current_class_name = class_names[label_val] # class_names are Low, Med, High; label_val is 0,1,2
            elif model_output_name == 'pca_type_output':
                current_class_name = class_names[label_val] # class_names are Red, White; label_val is 0,1
            else: # Normal quality
                current_class_name = class_names[label_val] # class_names are quality_3 etc, label_val is 0-6

            scatter_legend_handles.append(
                plt.Line2D([0], [0], marker='o', color='w', label=current_class_name,
                           markerfacecolor=legend_cmap_for_scatter_handles(i / (len(unique_scatter_labels) -1 if len(unique_scatter_labels) > 1 else 1) ), markersize=10)
            )
        
        # Only add legend if handles were created
        if scatter_legend_handles:
             legend1 = ax.legend(handles=scatter_legend_handles, title="True Labels (Dots)", loc='upper right')
             ax.add_artist(legend1)


        # Legend for contourf regions (Predicted Background)
        # class_names here should correspond to the classes Z can take.
        # For type, Z is 0,1. class_names is ['Red (0)', 'White (1)']
        # For normal quality, Z is 0-6. class_names is ['quality_3', ..., 'quality_9']
        # For binned quality, Z is 0,1,2. class_names is ['Low', 'Medium', 'High']
        
        num_contour_legend_items = num_contour_classes
        contour_legend_class_names = class_names # This should be the appropriate set of names

        contour_legend_handles = []
        for i in range(num_contour_legend_items):
            contour_legend_handles.append(
                plt.Rectangle((0, 0), 1, 1, 
                              facecolor=cmap_light(i / (num_contour_legend_items - 1 if num_contour_legend_items > 1 else 1)), 
                              label=contour_legend_class_names[i])
            )
        if contour_legend_handles:
            ax.legend(handles=contour_legend_handles, title="Predicted Regions", loc='lower left')

    plt.savefig(os.path.join(parent_dir, filename), bbox_inches='tight', dpi=200)
    print(f"Decision boundary plot saved to {os.path.join(parent_dir, filename)}")


# Main function to generate PCA visualizations
def generate_pca_visualizations(X_data_for_pca, y_type_all_data, 
                                y_quality_original_labels_all_data, 
                                feature_names_for_pca, # Will be original or encoded names
                                quality_target_names_from_onehot, 
                                parent_dir_path,
                                is_data_encoded=False): # Flag to know if data is encoded
    # 1. Perform PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca_transformed_all = pca.fit_transform(X_data_for_pca)

    # 2. Create PCA Information Tables
    create_pca_explained_variance_table(pca, parent_dir_path)
    create_pca_component_loadings_table(pca, feature_names_for_pca, parent_dir_path) # Use appropriate feature names

    y_quality_one_hot_all_data = pd.get_dummies(y_quality_original_labels_all_data, prefix='quality', dtype=int).values
    num_actual_quality_classes = y_quality_one_hot_all_data.shape[1]

    # Split PCA-transformed data for the PCA-specific model
    # Note: y_type_all_data and y_quality_one_hot_all_data are based on the full original dataset labels
    X_train_pca, X_test_pca, \
    y_train_type_for_pca_model, y_test_type_for_pca_model, \
    y_train_quality_one_hot_for_pca_model, _, \
    _, y_test_quality_labels_for_pca_model = train_test_split(
        X_pca_transformed_all, 
        y_type_all_data, 
        y_quality_one_hot_all_data, 
        y_quality_original_labels_all_data, 
        test_size=0.2, random_state=42,
        stratify=y_type_all_data # Stratify by type for balance if possible
    )
    
    # This model is trained on 2D PCA components derived from either original or encoded data
    pca_model = train_pca_model(X_train_pca, y_train_type_for_pca_model, 
                                y_train_quality_one_hot_for_pca_model, num_actual_quality_classes)

    min_original_quality_score = y_quality_original_labels_all_data.min()
    
    pca_plot_title_suffix = "(PCA on Encoded Features)" if is_data_encoded else "(PCA on Original Features)"

    fig_type, ax_type = plt.subplots(figsize=(8, 6))
    plot_decision_boundary(ax_type, pca_model, X_test_pca, y_test_type_for_pca_model,
                           'pca_type_output', f'Wine Type Decision Boundary {pca_plot_title_suffix}',
                           ['Red (0)', 'White (1)'], parent_dir_path, f'decision_boundary_type_pca{"_encoded" if is_data_encoded else ""}.png')
    plt.close(fig_type)

    fig_quality_norm, ax_quality_norm = plt.subplots(figsize=(10, 8))
    y_test_quality_indices_scatter = [q - min_original_quality_score for q in y_test_quality_labels_for_pca_model]
    plot_decision_boundary(ax_quality_norm, pca_model, X_test_pca, y_test_quality_indices_scatter,
                           'pca_quality_output', f'Wine Quality Decision Boundary {pca_plot_title_suffix}',
                           quality_target_names_from_onehot, 
                           parent_dir_path, f'decision_boundary_quality_pca{"_encoded" if is_data_encoded else ""}.png',
                           min_original_quality_score_for_binning=min_original_quality_score)
    plt.close(fig_quality_norm)

# funciones que se ejecutan al finalizar los for para hacer los plots de acc para todos los modelos
def plot_accuracy_heatmap(model_metrics_list, l1_neuron_values, l2_neuron_values, metric_key, metric_name_display, output_dir):
    """
    Generates and saves a heatmap of model accuracies.
    Args:
        model_metrics_list: List of dictionaries, each containing 'l1', 'l2', and metric_key.
        l1_neuron_values: List of L1 neuron counts (e.g., range(1, 21)).
        l2_neuron_values: List of L2 neuron counts (e.g., range(1, 21)).
        metric_key: The key in model_metrics_list dicts corresponding to the accuracy to plot.
        metric_name_display: String name for the metric for titles/labels (e.g., "Type Accuracy").
        output_dir: Directory to save the heatmap image.
    """
    num_l1_options = len(l1_neuron_values)
    num_l2_options = len(l2_neuron_values)
    accuracies_matrix = np.zeros((num_l1_options, num_l2_options))

    for item in model_metrics_list:
        l1 = item['l1']
        l2 = item['l2']
        accuracy = item[metric_key]
        try:
            idx_l1 = l1_neuron_values.index(l1)
            idx_l2 = l2_neuron_values.index(l2)
            accuracies_matrix[idx_l1, idx_l2] = accuracy
        except ValueError:
            print(f"Warning: Neuron configuration L1={l1}, L2={l2} not found in provided neuron value lists for heatmap.")
            continue


    plt.figure(figsize=(14, 12))
    sns.heatmap(accuracies_matrix, annot=True, fmt=".3f", cmap="viridis",
                xticklabels=l2_neuron_values, yticklabels=l1_neuron_values,
                cbar_kws={'label': metric_name_display})
    plt.xlabel("Number of Neurons in Hidden Layer 2")
    plt.ylabel("Number of Neurons in Hidden Layer 1")
    plt.title(f"Heatmap of Model {metric_name_display}\nvs. Neuron Configuration")
    
    heatmap_filename = f"accuracy_heatmap_{metric_key}_L1_vs_L2.png"
    heatmap_path = os.path.join(output_dir, heatmap_filename)
    plt.savefig(heatmap_path, bbox_inches='tight')
    plt.close()
    print(f"{metric_name_display} heatmap saved to: {heatmap_path}")

def plot_accuracy_line_graph(model_metrics_list, metric_key, metric_name_display, best_metric_config, output_dir):
    """
    Generates and saves a line graph of model accuracies across configurations.
    Args:
        model_metrics_list: List of dictionaries, each containing 'l1', 'l2', and metric_key.
        metric_key: The key in model_metrics_list dicts corresponding to the accuracy to plot.
        metric_name_display: String name for the metric for titles/labels.
        best_metric_config: Dictionary {'l1': best_l1, 'l2': best_l2, 'value': best_acc_value} for the metric.
        output_dir: Directory to save the line graph image.
    """
    if not model_metrics_list:
        print(f"No data to plot for {metric_name_display} line graph.")
        return

    # Sort by L1 then L2 for a somewhat ordered plot
    sorted_metrics = sorted(model_metrics_list, key=lambda x: (x['l1'], x['l2']))
    
    config_labels = [f"L1:{item['l1']},L2:{item['l2']}" for item in sorted_metrics]
    accuracies_to_plot = [item[metric_key] for item in sorted_metrics]

    plt.figure(figsize=(max(25, len(config_labels) * 0.075), 10)) # Dynamic width
    plt.plot(config_labels, accuracies_to_plot, marker='o', linestyle='-', label=metric_name_display, markersize=3, linewidth=0.8)

    if best_metric_config and best_metric_config['value'] > -1:
        best_config_label = f"L1:{best_metric_config['l1']},L2:{best_metric_config['l2']}"
        try:
            best_config_idx = config_labels.index(best_config_label)
            plt.plot(best_config_idx, best_metric_config['value'], marker='*', markersize=15, color='red',
                     label=f'Best {metric_name_display}: {best_metric_config["value"]:.4f} ({best_config_label})')
        except ValueError:
            print(f"Warning: Best configuration label '{best_config_label}' for {metric_name_display} not found in config_labels for line plot.")

    plt.xlabel("Model Configuration (Neurons L1, Neurons L2)")
    plt.ylabel(metric_name_display)
    plt.title(f"{metric_name_display} of Model Configurations")
    plt.xticks(rotation=90, ha='right', fontsize=min(8, 400 / len(config_labels) if len(config_labels) > 0 else 8) ) # Adjust fontsize

    min_acc_val = min(accuracies_to_plot) if accuracies_to_plot else 0.0
    max_acc_val = max(accuracies_to_plot) if accuracies_to_plot else 1.0
    y_tick_step = 0.05 if (max_acc_val - min_acc_val) > 0.2 else 0.01
    
    # Ensure y_ticks_start is not greater than min_acc_val
    y_ticks_start = (np.floor(min_acc_val / y_tick_step) * y_tick_step) if accuracies_to_plot else 0.0
    if y_ticks_start > min_acc_val and accuracies_to_plot: # Adjust if floor was too high
         y_ticks_start = max(0, y_ticks_start - y_tick_step)


    y_ticks_end = (np.ceil(max_acc_val / y_tick_step) * y_tick_step + y_tick_step) if accuracies_to_plot else 1.0
    if y_ticks_end < max_acc_val + y_tick_step and accuracies_to_plot: # Adjust if ceil was too low
        y_ticks_end = max_acc_val + y_tick_step


    # Ensure y_ticks_start is less than y_ticks_end before creating arange
    if y_ticks_start < y_ticks_end :
        plt.yticks(np.arange(y_ticks_start, y_ticks_end, y_tick_step))
    else: # Fallback if range is too small or problematic
        plt.yticks(np.linspace(min_acc_val, max_acc_val, num=10 if max_acc_val > min_acc_val else 2))


    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()

    line_plot_filename = f"accuracy_line_graph_{metric_key}.png"
    line_plot_path = os.path.join(output_dir, line_plot_filename)
    plt.savefig(line_plot_path)
    plt.close()
    print(f"{metric_name_display} line graph saved to: {line_plot_path}")


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, 'functional_api', f'functional_api_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
os.makedirs(parent_dir, exist_ok=True)

# Cargar el archivo Excel
white_wine_data_path = os.path.join(current_dir, 'winequality-white.csv')
red_wine_data_path = os.path.join(current_dir, 'winequality-red.csv')

white_wine_data = pd.read_csv(white_wine_data_path, sep=';')
red_wine_data = pd.read_csv(red_wine_data_path, sep=';')
# Agregar la columna 'type' a cada DataFrame
white_wine_data['type'] = 1 # 1 for white
red_wine_data['type'] = 0 # 0 for red

# Juntar los dataframes
combined_data = pd.concat([white_wine_data, red_wine_data], axis=0)
print(combined_data.head())
print(combined_data.tail())

# Defino las x y las y
feature_columns = [col for col in combined_data.columns if col not in ['type', 'quality']]
X_original_features = combined_data[feature_columns]
y_labels = combined_data[['type', 'quality']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_original_features)

y_one_hot_labels = pd.get_dummies(y_labels, columns=['quality'], prefix='quality', dtype=int)

# Split data BEFORE autoencoder training to prevent leakage from test set into AE training
X_train_orig_scaled, X_test_orig_scaled, y_train, y_test = train_test_split(
    X_scaled, y_one_hot_labels, test_size=0.2, random_state=42
)

# --- Autoencoder Training ---
original_dim = X_train_orig_scaled.shape[1]
encoding_dim = int(original_dim / 2) # Example: reduce dimensionality by half, or choose a fixed number e.g. 7
# encoding_dim = 7 # Alternative fixed encoding dimension

# Further split X_train_orig_scaled for autoencoder's own validation set
X_train_for_ae, X_val_for_ae = train_test_split(X_train_orig_scaled, test_size=0.2, random_state=42)

encoder_model, autoencoder_model, ae_history = build_and_train_autoencoder(
    X_train_for_ae, X_val_for_ae, original_dim, encoding_dim, epochs=50 
)
plot_autoencoder_loss(ae_history, parent_dir) # Plot AE loss

# Transform the main model's training and testing data using the trained encoder
X_train_encoded = encoder_model.predict(X_train_orig_scaled)
X_test_encoded = encoder_model.predict(X_test_orig_scaled)
print(f"Original X_train shape: {X_train_orig_scaled.shape}, Encoded X_train shape: {X_train_encoded.shape}")
print(f"Original X_test shape: {X_test_orig_scaled.shape}, Encoded X_test shape: {X_test_encoded.shape}")

# Prepare y_train and y_test for the main multi-output model (these are already from y_one_hot_labels split)
y_train_type = y_train['type']
y_train_quality = y_train[[col for col in y_train.columns if 'quality_' in col]]
y_test_type = y_test['type']
y_test_quality = y_test[[col for col in y_test.columns if 'quality_' in col]]

# --- Main Model Definition and Training using ENCODED features ---
input_layer = Input(shape=(encoding_dim,), name='input_layer') # Shape is now encoding_dim

all_model_metrics = []
best_metrics_info = {
    'type_accuracy': {'value': -1.0, 'l1': -1, 'l2': -1, 'description': ''},
    'quality_accuracy': {'value': -1.0, 'l1': -1, 'l2': -1, 'description': ''}
}

L1_NEURONS_START = 1
L1_NEURONS_END = 21 
L2_NEURONS_START = 1
L2_NEURONS_END = 21

L1_NEURONS_RANGE = range(L1_NEURONS_START, L1_NEURONS_END)
L2_NEURONS_RANGE = range(L2_NEURONS_START, L2_NEURONS_END)


# Hidden layers
for l1_neurons in L1_NEURONS_RANGE:
    for l2_neurons in L2_NEURONS_RANGE:
        
        model_description = f"L1_{l1_neurons}_L2_{l2_neurons}"
        print(f"\n--- Training and Evaluating Model: {model_description} ---")
        #set up directory for each model
        model_dir = os.path.join(parent_dir, f'model_{model_description}')
        os.makedirs(model_dir, exist_ok=True)
        
        hidden_1 = Dense(l1_neurons, activation='relu', name='hidden_layer_1')(input_layer)
        hidden_2 = Dense(l2_neurons, activation='relu', name='hidden_layer_2')(hidden_1)

        type_output = Dense(1, activation='sigmoid', name='type_output')(hidden_2)
        quality_output = Dense(y_train_quality.shape[1], activation='softmax', name='quality_output')(hidden_2)
        model = Model(inputs=input_layer, outputs=[type_output, quality_output])

        summary_path = os.path.join(model_dir, 'model_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"Model summary saved to {summary_path}")


        # Plot model architecture
        plot_model(model, to_file=os.path.join(model_dir, 'model_plot.png'), show_shapes=True, show_layer_names=True)

        # Compilo el modelo
        model.compile(optimizer=Adam(learning_rate=0.001),
                    loss={'type_output': 'binary_crossentropy', 'quality_output': 'categorical_crossentropy'},
                    metrics={'type_output': tf.keras.metrics.BinaryAccuracy(name='type_accuracy'), 
                             'quality_output': tf.keras.metrics.CategoricalAccuracy(name='quality_accuracy')})

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train the model
        print(f"Starting model training for {model_description}...")
        history = model.fit(X_train_encoded, # Use encoded data
                            {'type_output': y_train_type, 'quality_output': y_train_quality},
                            epochs=100,
                            batch_size=64,
                            validation_data=(X_test_encoded, {'type_output': y_test_type, 'quality_output': y_test_quality}), # Use encoded data
                            callbacks=[early_stopping],
                            verbose=1)
        print(f"Model training finished for {model_description}.")

        print(f"Evaluating {model_description} on test data...")
        eval_results_dict = model.evaluate(X_test_encoded, {'type_output': y_test_type, 'quality_output': y_test_quality}, verbose=0, return_dict=True) # Use encoded
        print(f"DEBUG: eval_results_dict for {model_description}: {eval_results_dict}") 
        
        test_type_accuracy = eval_results_dict.get('type_output_type_accuracy', eval_results_dict.get('type_accuracy', 0.0))
        test_quality_accuracy = eval_results_dict.get('quality_output_quality_accuracy', eval_results_dict.get('quality_accuracy', 0.0))
        test_total_loss = eval_results_dict.get('loss', 0.0)
        
        print(f"{model_description} - Total Loss: {test_total_loss:.4f}")
        print(f"{model_description} - Test Type Accuracy: {test_type_accuracy:.4f}, Test Quality Accuracy: {test_quality_accuracy:.4f}")

        all_model_metrics.append({
            'l1': l1_neurons,
            'l2': l2_neurons,
            'type_accuracy': test_type_accuracy,
            'quality_accuracy': test_quality_accuracy,
            'val_loss': history.history['val_loss'][early_stopping.best_epoch if hasattr(early_stopping, 'best_epoch') else -1]
        })

        if test_type_accuracy > best_metrics_info['type_accuracy']['value']:
            best_metrics_info['type_accuracy'].update({'value': test_type_accuracy, 'l1': l1_neurons, 'l2': l2_neurons, 'description': model_description})
        
        if test_quality_accuracy > best_metrics_info['quality_accuracy']['value']:
            best_metrics_info['quality_accuracy'].update({'value': test_quality_accuracy, 'l1': l1_neurons, 'l2': l2_neurons, 'description': model_description})

        quality_cols_for_report = [col for col in y_train_quality.columns]
        plot_loss_and_accuracy(history, model_dir)
        save_classification_reports(model, X_test_encoded, y_test_type, y_test_quality, quality_cols_for_report, model_dir) # Use encoded
        plot_confusion_matrices(model, X_test_encoded, y_test_type, y_test_quality, quality_cols_for_report, model_dir)   # Use encoded
        save_quality_classification_report_with_tolerance(model, X_test_encoded, y_test_quality, quality_cols_for_report, model_dir) # Use encoded
        plot_quality_confusion_matrix_with_tolerance(model, X_test_encoded, y_test_quality, quality_cols_for_report, model_dir) # Use encoded

        # PCA visualizations will now run on ENCODED data
        # Encode the full X_scaled dataset for PCA visualization purposes
        X_scaled_encoded_for_pca = encoder_model.predict(X_scaled)
        encoded_feature_names = [f'encoded_dim_{i}' for i in range(encoding_dim)]
        
        y_type_full_dataset = combined_data['type'].values # From original combined_data
        y_quality_original_labels_full_dataset = combined_data['quality'].values # From original combined_data

        print(f"\nStarting PCA and Decision Boundary Visualizations for {model_description} (on encoded features)...")
        generate_pca_visualizations(
            X_scaled_encoded_for_pca, # Pass encoded full dataset
            y_type_full_dataset,
            y_quality_original_labels_full_dataset,
            encoded_feature_names, # Pass names for encoded dimensions
            quality_cols_for_report, 
            model_dir,
            is_data_encoded=True # Flag that data is encoded
        )
        print(f"PCA and Decision Boundary Visualizations finished for {model_description}.")

# --- End of loops ---

# After all models are trained and evaluated, generate summary plots
print("\n--- Generating Overall Accuracy Summary Plots ---")
l1_neuron_options_list = list(L1_NEURONS_RANGE)
l2_neuron_options_list = list(L2_NEURONS_RANGE)

if all_model_metrics:
    plot_accuracy_heatmap(all_model_metrics, l1_neuron_options_list, l2_neuron_options_list,
                          'type_accuracy', f'Type Accuracy (AE EncDim {encoding_dim}, Test Set)', parent_dir)
    plot_accuracy_heatmap(all_model_metrics, l1_neuron_options_list, l2_neuron_options_list,
                          'quality_accuracy', f'Quality Accuracy (AE EncDim {encoding_dim}, Test Set)', parent_dir)

    plot_accuracy_line_graph(all_model_metrics, 'type_accuracy', f'Type Accuracy (AE EncDim {encoding_dim}, Test Set)',
                             best_metrics_info['type_accuracy'], parent_dir)
    plot_accuracy_line_graph(all_model_metrics, 'quality_accuracy', f'Quality Accuracy (AE EncDim {encoding_dim}, Test Set)',
                             best_metrics_info['quality_accuracy'], parent_dir)

    print("\n--- Overall Best Model Configurations (with Autoencoder) ---")
    print(f"Best Type Accuracy ({best_metrics_info['type_accuracy']['value']:.4f}) achieved with model: {best_metrics_info['type_accuracy']['description']}")
    print(f"Best Quality Accuracy ({best_metrics_info['quality_accuracy']['value']:.4f}) achieved with model: {best_metrics_info['quality_accuracy']['description']}")
    best_val_loss_model = min(all_model_metrics, key=lambda x: x.get('val_loss', float('inf')))
    print(f"Model with Best Validation Loss ({best_val_loss_model.get('val_loss', float('inf')):.4f}): {best_val_loss_model['l1']}_{best_val_loss_model['l2']} (AE EncDim {encoding_dim})")
else:
    print("No model metrics collected, skipping summary plots.")

plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')