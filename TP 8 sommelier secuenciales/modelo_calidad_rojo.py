import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import datetime
from tensorflow.keras.utils import to_categorical



def plot_proportions(df, target_col, directory):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=target_col, palette=['red', 'yellow'])
    plt.title(f'Proportions of {target_col}')
    plt.xlabel(target_col)
    plt.ylabel('Count')
    plt.savefig(os.path.join(directory, f'proportions_of_{target_col}.png'))

def plot_pca(df, feature_cols, target_col, directory):
    """
    Performs PCA on the given features, plots the first two components,
    and saves tables for explained variance and component loadings.
    Args:
        df: Pandas DataFrame containing the data.
        feature_cols: List of column names to use as features for PCA.
        target_col: Name of the target column for coloring the scatter plot.
        directory: The directory to save the plots and tables in.
    """
    print(f"\n--- Generating PCA plots and tables for features: {', '.join(feature_cols)} ---")
    print(f"Output directory for PCA analysis: {directory}")

    # 1. Prepare Data
    X_pca_raw = df[feature_cols].values
    y_pca = df[target_col].values

    # 2. Apply PCA
    pca = PCA(n_components=2, random_state=42) # random_state for reproducibility
    X_pca_transformed = pca.fit_transform(X_pca_raw)

    # Create a base DataFrame for PCA plots, including the original target column for coloring/binning
    pca_plot_df = pd.DataFrame(data=X_pca_transformed, columns=['Principal Component 1', 'Principal Component 2'])
    # Ensure original target values are present for both plotting types
    # It's assumed df's index aligns or .values handles any potential mismatch with pca_plot_df's default index
    pca_plot_df[target_col] = df[target_col].values


    # --- Plot 1: Original Quality Categories (Viridis Palette) ---
    # This plot is similar to the original PCA scatter plot logic
    plt.figure(figsize=(10, 7))

    # Use a temporary string column from the original numeric target for hue, to ensure discrete colors and legend
    temp_hue_col_original = target_col + "_str_hue_original"
    pca_plot_df[temp_hue_col_original] = pca_plot_df[target_col].astype(str)

    unique_hue_values_str_original = sorted(pca_plot_df[temp_hue_col_original].unique())
    palette_map_original = {}
    legend_labels_map_original = {}

    if unique_hue_values_str_original:
        qualities_original_numeric = sorted([int(q_str) for q_str in unique_hue_values_str_original])
        num_qualities_original = len(qualities_original_numeric)
        colors_original = sns.color_palette("viridis", n_colors=max(num_qualities_original, 2)) # Ensure at least 2 colors
        for i, quality_val_numeric in enumerate(qualities_original_numeric):
            quality_str = str(quality_val_numeric)
            palette_map_original[quality_str] = colors_original[i]
            legend_labels_map_original[quality_str] = f'Quality {quality_val_numeric}'

    scatter_plot_ax_original = sns.scatterplot(
        x='Principal Component 1',
        y='Principal Component 2',
        hue=temp_hue_col_original, # Use the stringified original quality for hue
        data=pca_plot_df,
        palette=palette_map_original,
        alpha=0.7,
        edgecolor='k',
        linewidth=0.5
    )
    plt.title(f'PCA of Wine Data ({len(feature_cols)} Features) by Quality (Original Categories)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    handles_orig, current_labels_str_from_plot_orig = scatter_plot_ax_original.get_legend_handles_labels()
    new_plot_labels_orig = [legend_labels_map_original.get(lbl_str, lbl_str) for lbl_str in current_labels_str_from_plot_orig]
    if handles_orig and new_plot_labels_orig:
        plt.legend(handles_orig, new_plot_labels_orig, title='Wine Quality')
    elif handles_orig: # Fallback if mapping somehow fails
        plt.legend(handles_orig, current_labels_str_from_plot_orig, title='Wine Quality')

    original_scatter_plot_path = os.path.join(directory, 'pca_scatter_plot_original_categories.png')
    plt.savefig(original_scatter_plot_path, bbox_inches='tight')
    plt.close()
    print(f"PCA scatter plot (original categories) saved to: {original_scatter_plot_path}")

    # Clean up the temporary column used for the original plot if it might conflict, though it shouldn't here.
    # pca_plot_df.drop(columns=[temp_hue_col_original], inplace=True, errors='ignore')


    # --- Plot 2: Binned Quality Categories (Red, Yellow, Green) ---
    plt.figure(figsize=(10, 7))

    # Define a function to map quality to bins (using the numeric target_col)
    def map_quality_to_bin(quality):
        if quality <= 4: return 'Quality 3-4'
        elif quality <= 7: return 'Quality 5-7'
        else: return 'Quality 8-9'
            
    pca_plot_df['quality_bin'] = pca_plot_df[target_col].apply(map_quality_to_bin)
    
    hue_order_binned = ['Quality 3-4', 'Quality 5-7', 'Quality 8-9']
    palette_colors_binned = {
        'Quality 3-4': 'red',
        'Quality 5-7': 'yellow',
        'Quality 8-9': 'green'
    }
    
    scatter_plot_ax_binned = sns.scatterplot(
        x='Principal Component 1',
        y='Principal Component 2',
        hue='quality_bin', # Use the new binned column
        data=pca_plot_df,
        palette=palette_colors_binned,
        hue_order=hue_order_binned, # Ensure correct legend order
        alpha=0.7,
        edgecolor='k',
        linewidth=0.5
    )
    plt.title(f'PCA of Wine Data ({len(feature_cols)} Features) by Quality (Binned)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Wine Quality Bins') # Updated legend title
    
    binned_scatter_plot_path = os.path.join(directory, 'pca_scatter_plot_binned.png')
    plt.savefig(binned_scatter_plot_path, bbox_inches='tight')
    plt.close()
    print(f"PCA scatter plot (binned categories) saved to: {binned_scatter_plot_path}")


    # 4. Explained Variance Table
    explained_variance = pca.explained_variance_ratio_
    explained_variance_df_data = {
        'Principal Component': ['PC1', 'PC2'],
        'Variance Explained Ratio': [f"{var:.4f}" for var in explained_variance] # Format as string
    }
    explained_variance_df = pd.DataFrame(explained_variance_df_data)
    
    total_variance = explained_variance.sum()
    total_row = pd.DataFrame({
        'Principal Component': ['Total Explained (PC1+PC2)'],
        'Variance Explained Ratio': [f"{total_variance:.4f}"]
    })
    explained_variance_df = pd.concat([explained_variance_df, total_row], ignore_index=True)

    fig_ev, ax_ev = plt.subplots(figsize=(6, max(2.5, explained_variance_df.shape[0] * 0.5)))
    ax_ev.axis('tight'); ax_ev.axis('off')
    ev_table = ax_ev.table(cellText=explained_variance_df.values,
                           colLabels=explained_variance_df.columns,
                           cellLoc='center', loc='center', colWidths=[0.5, 0.5])
    ev_table.auto_set_font_size(False); ev_table.set_fontsize(10); ev_table.scale(1.1, 1.1)
    plt.title("PCA Explained Variance Ratio", pad=20, fontsize=12)
    explained_variance_path = os.path.join(directory, 'pca_explained_variance_table.png')
    plt.savefig(explained_variance_path, bbox_inches='tight')
    plt.close(fig_ev)
    print(f"PCA explained variance table saved to: {explained_variance_path}")
    print("\nPCA Explained Variance Ratio:\n", explained_variance_df)

    # 5. PCA Components (Loadings) Table
    loadings = pca.components_ 
    loadings_df = pd.DataFrame(loadings.T, columns=['PC1 Loading', 'PC2 Loading'], index=feature_cols)
    
    fig_load, ax_load = plt.subplots(figsize=(8, max(4, len(feature_cols) * 0.4 + 1)))
    ax_load.axis('tight'); ax_load.axis('off')

    cell_text_loadings = []
    for r_idx in loadings_df.index: # Iterate through original feature names
        row_values = [f"{val:.4f}" for val in loadings_df.loc[r_idx]]
        cell_text_loadings.append(row_values)

    load_table = ax_load.table(cellText=cell_text_loadings,
                               rowLabels=loadings_df.index, # Original feature names as row labels
                               colLabels=loadings_df.columns, # PC1 Loading, PC2 Loading
                               cellLoc='center', loc='center', colWidths=[0.3, 0.3])
    load_table.auto_set_font_size(False); load_table.set_fontsize(10); load_table.scale(1.1, 1.1)
    plt.title(f"PCA Component Loadings ({len(feature_cols)} Features)", pad=20, fontsize=12)
    loadings_path = os.path.join(directory, 'pca_component_loadings_table.png')
    plt.savefig(loadings_path, bbox_inches='tight')
    plt.close(fig_load)
    print(f"PCA component loadings table saved to: {loadings_path}")
    print("\nPCA Component Loadings:\n", loadings_df.round(4))

def scatterplot_vs_quality(df, x_col, y_col, output_dir):
    # --- Plot 1: Binned Qualities ---
    plt.figure(figsize=(10, 6))
    df_plot_binned = df.copy() # Use a copy for binned plot modifications

    def map_quality_to_bin(quality):
        if quality <= 4: return 'Quality 3-4'
        elif quality <= 7: return 'Quality 5-7'
        else: return 'Quality 8-9'
            
    df_plot_binned['quality_bin'] = df_plot_binned['quality'].apply(map_quality_to_bin)
    
    hue_order_binned = ['Quality 3-4', 'Quality 5-7', 'Quality 8-9']
    palette_colors_binned = {
        'Quality 3-4': 'red',
        'Quality 5-7': 'yellow',
        'Quality 8-9': 'green'
    }
    
    sns.scatterplot(data=df_plot_binned, x=x_col, y=y_col, hue='quality_bin', palette=palette_colors_binned, hue_order=hue_order_binned)
    plt.title(f'Scatter Plot of {x_col} vs {y_col} by Wine Quality (Binned)')
    plt.xlabel(x_col); plt.ylabel(y_col)
    plt.legend(title='Wine Quality Bins')
    binned_plot_filename = f'scatterplot_vs_quality_binned_{x_col.replace(" ", "_").lower()}_vs_{y_col.replace(" ", "_").lower()}.png'
    plt.savefig(os.path.join(output_dir, binned_plot_filename))
    plt.close()
    print(f"Binned scatter plot saved to: {os.path.join(output_dir, binned_plot_filename)}")

    # --- Plot 2: Original Quality Categories ---
    plt.figure(figsize=(10, 6))
    
    # Determine unique qualities and create a palette
    unique_qualities = sorted(df['quality'].unique())
    num_unique_qualities = len(unique_qualities)
    
    # Using 'viridis' palette, similar to PCA plot, ensuring enough colors
    # If there are very few unique qualities, this still works.
    # If there are many, viridis is a good choice for ordered data.
    palette_original = sns.color_palette("viridis", n_colors=max(num_unique_qualities, 2)) # Ensure at least 2 colors for palette generation
    
    # Create a mapping from quality value to color for consistent legend if needed,
    # though seaborn should handle it well with a numeric hue and a list palette.
    # hue_order_original = unique_qualities # Optional: enforce order
    
    sns.scatterplot(data=df, x=x_col, y=y_col, hue='quality', palette=palette_original, hue_order=unique_qualities)
    plt.title(f'Scatter Plot of {x_col} vs {y_col} by Wine Quality (Original Categories)')
    plt.xlabel(x_col); plt.ylabel(y_col)
    plt.legend(title='Wine Quality') # Standard legend title
    original_plot_filename = f'scatterplot_vs_quality_original_{x_col.replace(" ", "_").lower()}_vs_{y_col.replace(" ", "_").lower()}.png'
    plt.savefig(os.path.join(output_dir, original_plot_filename))
    plt.close()
    print(f"Original categories scatter plot saved to: {os.path.join(output_dir, original_plot_filename)}")

def plot_decision_boundary(X_plot, y_plot, model_to_predict_with, scaler_original, feature_names_plot, directory, data_split_name, model_name_prefix, pca_transformer=None):
    plt.figure(figsize=(10, 6))

    # X_plot are the 2D points to be scattered (either original 2D features or 2D PCA-transformed features)
    x_min_data, x_max_data = X_plot[:, 0].min(), X_plot[:, 0].max()
    x_range_data = x_max_data - x_min_data
    x_margin = x_range_data * 0.05 if x_range_data > 1e-6 else 0.05 # 5% padding or small fixed
    x_min, x_max = x_min_data - x_margin, x_max_data + x_margin
    if x_max <= x_min + 1e-7: # Fallback for x-axis
        x_center = (x_min_data + x_max_data) / 2.0
        x_min = x_center - 0.1 # Ensure some width
        x_max = x_center + 0.1
        if x_max <= x_min + 1e-7: x_max = x_min + 0.2


    # Adjust y-axis for a tighter fit based on its data range, similar to seaborn's default
    y_data_min_actual = X_plot[:, 1].min()
    y_data_max_actual = X_plot[:, 1].max()
    y_data_range_actual = y_data_max_actual - y_data_min_actual
    
    if y_data_range_actual < 1e-6:  # Effectively zero range (e.g. all points on a horizontal line)
        # Use a small, fixed absolute margin to give some visual height.
        # This value (0.0025) means total height will be 0.005.
        margin_y = 0.0025
    else:
        # For non-zero range, use 5% of the actual data range as margin on each side.
        padding_percentage = 0.01
        margin_y = y_data_range_actual * padding_percentage
                                            
    y_min = y_data_min_actual - margin_y
    y_max = y_data_max_actual + margin_y
    
    # Fallback if y_min and y_max are still too close for np.linspace after calculations.
    # This handles cases where data range was tiny, making margin_y also tiny.
    if y_max <= y_min + 1e-7: # Epsilon for linspace distinctiveness
        # Ensure a minimal visible height for the plot.
        min_fallback_total_height = 0.02 # desired minimum total y-axis span
        y_center = (y_data_min_actual + y_data_max_actual) / 2.0
        y_min = y_center - (min_fallback_total_height / 2.0)
        y_max = y_center + (min_fallback_total_height / 2.0)
        if y_max <= y_min + 1e-7: # If still an issue (e.g. y_center was problematic or all points identical)
             y_max = y_min + min_fallback_total_height # Absolute minimum span

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # grid_points_2d are the coordinates in the 2D plot space
    grid_points_2d = np.c_[xx.ravel(), yy.ravel()]
    
    grid_points_for_model_prediction = grid_points_2d
    if pca_transformer:
        # If PCA was used for visualization, inverse_transform the 2D grid points
        # back to the original high-dimensional space before prediction.
        # This assumes pca_transformer was fit on data with the same scaling as model_to_predict_with expects.
        print(f"Inverse transforming {grid_points_2d.shape[0]} grid points using PCA for model prediction...")
        grid_points_for_model_prediction = pca_transformer.inverse_transform(grid_points_2d)
    # else:
        # If not using PCA, and if original data fed to model was scaled by 'scaler_original',
        # then grid_points_2d (which are in the space of X_plot) might need scaling.
        # HOWEVER, since we assume input data is ALREADY scaled, and X_plot comes from that,
        # grid_points_2d are already in the "scaled" space if feature_names_plot were original features.
        # The 'scaler_original' argument is not actively used for transforming grid_points_for_model_prediction here.
        # If X_plot was direct features (assumed scaled), grid_points_2d are in that space.

    # Predict using the main model (model_to_predict_with)
    # For multi-class, predict returns probabilities for each class
    Z_proba = model_to_predict_with.predict(grid_points_for_model_prediction)
    Z = np.argmax(Z_proba, axis=1) # Get the class with the highest probability
    Z = Z.reshape(xx.shape)

    # Define the full range of mapped quality classes (0 to 6)
    min_quality_mapped = 0
    max_quality_mapped = 6

    # Use fixed levels and vmin/vmax for consistent coloring across plots
    contour_levels = np.arange(min_quality_mapped - 0.5, max_quality_mapped + 1.5, 1) # e.g., -0.5, 0.5, ..., 6.5

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.viridis, levels=contour_levels, vmin=min_quality_mapped, vmax=max_quality_mapped)
    # Scatter the provided X_plot points (these are already in 2D)
    # y_plot should be 0-6 mapped labels for coloring
    scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, cmap=plt.cm.viridis, edgecolors='k', vmin=min_quality_mapped, vmax=max_quality_mapped)
    plt.xlabel(feature_names_plot[0])
    plt.ylabel(feature_names_plot[1])
    
    title = f'{model_name_prefix}: Decision Boundary ({data_split_name}) - {feature_names_plot[0]} vs {feature_names_plot[1]}'
    plt.title(title)
    
    # Updated legend for multi-class quality (y_plot is 0-6, corresponding to quality 3-9)
    try:
        # Create a colorbar if more than 1 class, or a simple legend if few distinct classes.
        # For many classes, a colorbar is often better with contourf.
        # Let's try to make a legend from scatter points if possible.
        unique_y_plot_vals = np.unique(y_plot)
        if len(unique_y_plot_vals) > 1 and len(unique_y_plot_vals) <= 10: # Arbitrary limit for too many legend entries
            handles, _ = scatter.legend_elements(prop="colors", num=len(unique_y_plot_vals))
            # Ensure legend labels are sorted by the actual quality value for correct mapping
            # The unique_y_plot_vals are already sorted (0-6)
            legend_labels = [f"Quality {int(q)+3}" for q in unique_y_plot_vals]
            if handles and legend_labels and len(handles) == len(legend_labels):
                 plt.legend(handles, legend_labels, title="Wine Quality")
            else: # Fallback if legend_elements doesn't behave as expected
                cbar = plt.colorbar(scatter, ticks=np.arange(min_quality_mapped, max_quality_mapped + 1))
                cbar.set_label('Wine Quality')
                cbar.set_ticklabels([str(q+3) for q in range(min_quality_mapped, max_quality_mapped + 1)])

        elif len(unique_y_plot_vals) == 1 : # Single class in plot
             handles, _ = scatter.legend_elements(prop="colors", num=1)
             legend_labels = [f"Quality {int(unique_y_plot_vals[0]) + 3}"]
             if handles and legend_labels:
                 plt.legend(handles, legend_labels, title="Wine Quality")
        else: # Fallback for many classes, or if legend_elements is tricky or no points
            cbar = plt.colorbar(scatter, ticks=np.arange(min_quality_mapped, max_quality_mapped + 1))
            cbar.set_label('Wine Quality')
            cbar.set_ticklabels([str(q+3) for q in range(min_quality_mapped, max_quality_mapped + 1)])


    except Exception as e:
        print(f"Could not generate legend/colorbar for decision boundary: {e}")

    
    filename_data_split_part = data_split_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    filename_model_prefix_part = model_name_prefix.lower().replace(" ", "_") # Already lowercased by convention
    save_filename = f'{filename_model_prefix_part}_decision_boundary_{filename_data_split_part}_{feature_names_plot[0].lower().replace(" ", "_")}_vs_{feature_names_plot[1].lower().replace(" ", "_")}.png'
    save_path = os.path.join(directory, save_filename)
    plt.savefig(save_path)
    print(f"Decision boundary plot saved to: {save_path}")
    plt.close()

def plot_model_metrics(model, X_test_scaled, y_test, X_train_scaled, y_train, X_full_scaled, y_full, feature_names, directory, model_name_prefix):
    """
    Generates and saves tables for classification metrics and model coefficients.
    Also plots and saves a confusion matrix.
    Args:
        model: The trained Keras model.
        X_test_scaled: Scaled test features.
        y_test: Test labels.
        X_train_scaled: Scaled train features.
        y_train: Train labels.
        X_full_scaled: Scaled full features.
        y_full: Full labels.
        feature_names: List of feature names.
        directory: The directory to save the plots in.
        model_name_prefix: Prefix for filenames and titles (e.g., "Logistic Regression").
    """
    print(f"\nGenerating model evaluation metrics for {model_name_prefix}...")

    # --- Test Set Metrics ---
    print(f"\n--- {model_name_prefix} Metrics on Test Set ---")
    # 1. Get Predictions (for Classification Report - based on Test Set)
    y_pred_proba_test_for_report = model.predict(X_test_scaled)
    y_pred_test_for_report = np.argmax(y_pred_proba_test_for_report, axis=1)

    # 2. Classification Metrics Table (based on Test Set)
    quality_labels_for_report = [f'Quality {i}' for i in range(3, 10)] # Labels for display
    labels_param_for_report = list(range(7)) # Parameter for classification_report (0-6)

    report_dict_test = classification_report(y_test, y_pred_test_for_report, labels=labels_param_for_report, target_names=quality_labels_for_report, output_dict=True, zero_division=0)
    overall_accuracy_test = report_dict_test.pop('accuracy', None)
    report_df_test = pd.DataFrame(report_dict_test).transpose()
    report_df_test['accuracy'] = '--'
    accuracy_value_for_table_test = round(overall_accuracy_test, 4) if overall_accuracy_test is not None else '--'
    report_df_test.loc['accuracy'] = ['--', '--', '--', '--', accuracy_value_for_table_test]
    
    for col in ['precision', 'recall', 'f1-score']: 
        if col in report_df_test.columns:
            report_df_test[col] = report_df_test[col].apply(lambda x: round(x, 4) if isinstance(x, (float, int)) else x)
    if 'support' in report_df_test.columns:
        report_df_test['support'] = report_df_test['support'].apply(lambda x: int(x) if isinstance(x, (float, int)) and pd.notna(x) else x)

    fig_metrics_test, ax_metrics_test = plt.subplots(figsize=(11, 4.5)) 
    ax_metrics_test.axis('tight'); ax_metrics_test.axis('off')
    display_columns = ['precision', 'recall', 'f1-score', 'support', 'accuracy']
    final_display_columns_test = [col for col in display_columns if col in report_df_test.columns]
    cell_text_metrics_test = []
    for r_idx in report_df_test.index:
        row_values = []
        for c_idx in final_display_columns_test:
            val = report_df_test.loc[r_idx, c_idx]
            if isinstance(val, float):
                row_values.append("{:.4f}".format(val))
            elif c_idx == 'support' and isinstance(val, (int, np.integer)):
                row_values.append(str(val))
            else: 
                row_values.append(str(val))
        cell_text_metrics_test.append(row_values)

    table_metrics_test = ax_metrics_test.table(cellText=cell_text_metrics_test,
                                     colLabels=final_display_columns_test,
                                     rowLabels=report_df_test.index,
                                     cellLoc='center', loc='center', colWidths=[0.18]*len(final_display_columns_test))
    table_metrics_test.auto_set_font_size(False); table_metrics_test.set_fontsize(10); table_metrics_test.scale(1.2, 1.2)
    plt.title(f"{model_name_prefix} Classification Metrics (Test Set)", pad=20) 
    metrics_path_test = os.path.join(directory, f'{model_name_prefix.lower().replace(" ", "_")}_classification_metrics_table_test_set.png') 
    plt.savefig(metrics_path_test, bbox_inches='tight'); plt.close(fig_metrics_test)
    print(f"Classification metrics table (Test Set) saved to: {metrics_path_test}")
    print(f"\nClassification Report for {model_name_prefix} (Test Set):\n", classification_report(y_test, y_pred_test_for_report, labels=labels_param_for_report, target_names=quality_labels_for_report, zero_division=0))

    # --- Full Data Set Metrics ---
    print(f"\n--- {model_name_prefix} Metrics on Full Data Set ---")
    y_pred_proba_full = model.predict(X_full_scaled)
    y_pred_full_for_report = np.argmax(y_pred_proba_full, axis=1)

    report_dict_full = classification_report(y_full, y_pred_full_for_report, labels=labels_param_for_report, target_names=quality_labels_for_report, output_dict=True, zero_division=0)
    overall_accuracy_full = report_dict_full.pop('accuracy', None)
    report_df_full = pd.DataFrame(report_dict_full).transpose()
    report_df_full['accuracy'] = '--'
    accuracy_value_for_table_full = round(overall_accuracy_full, 4) if overall_accuracy_full is not None else '--'
    report_df_full.loc['accuracy'] = ['--', '--', '--', '--', accuracy_value_for_table_full]

    for col in ['precision', 'recall', 'f1-score']:
        if col in report_df_full.columns:
            report_df_full[col] = report_df_full[col].apply(lambda x: round(x, 4) if isinstance(x, (float, int)) else x)
    if 'support' in report_df_full.columns:
        report_df_full['support'] = report_df_full['support'].apply(lambda x: int(x) if isinstance(x, (float, int)) and pd.notna(x) else x)
    
    fig_metrics_full, ax_metrics_full = plt.subplots(figsize=(11, 4.5))
    ax_metrics_full.axis('tight'); ax_metrics_full.axis('off')
    final_display_columns_full = [col for col in display_columns if col in report_df_full.columns]
    cell_text_metrics_full = []
    for r_idx in report_df_full.index:
        row_values = []
        for c_idx in final_display_columns_full:
            val = report_df_full.loc[r_idx, c_idx]
            if isinstance(val, float):
                row_values.append("{:.4f}".format(val))
            elif c_idx == 'support' and isinstance(val, (int, np.integer)):
                row_values.append(str(val))
            else:
                row_values.append(str(val))
        cell_text_metrics_full.append(row_values)

    table_metrics_full = ax_metrics_full.table(cellText=cell_text_metrics_full,
                                     colLabels=final_display_columns_full,
                                     rowLabels=report_df_full.index,
                                     cellLoc='center', loc='center', colWidths=[0.18]*len(final_display_columns_full))
    table_metrics_full.auto_set_font_size(False); table_metrics_full.set_fontsize(10); table_metrics_full.scale(1.2, 1.2)
    plt.title(f"{model_name_prefix} Classification Metrics (Full Data)", pad=20)
    metrics_path_full = os.path.join(directory, f'{model_name_prefix.lower().replace(" ", "_")}_classification_metrics_table_full_data.png')
    plt.savefig(metrics_path_full, bbox_inches='tight'); plt.close(fig_metrics_full)
    print(f"Classification metrics table (Full Data) saved to: {metrics_path_full}")
    print(f"\nClassification Report for {model_name_prefix} (Full Data):\n", classification_report(y_full, y_pred_full_for_report, labels=labels_param_for_report, target_names=quality_labels_for_report, zero_division=0))

    # 3. Confusion Matrices (Test Set and Full Data)
    print(f"\n--- {model_name_prefix} Confusion Matrices ---")
    datasets_for_cm = {
        "Test Set": (X_test_scaled, y_test),
        "Full Data": (X_full_scaled, y_full)
    }

    for data_split_name, (X_data_scaled, y_data_true) in datasets_for_cm.items():
        y_pred_proba = model.predict(X_data_scaled)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        cm = confusion_matrix(y_data_true, y_pred, labels=list(range(7)))
        fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
        cm_display_labels = [f'Q{i+3}' for i in range(7)]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=cm_display_labels, 
                    yticklabels=cm_display_labels,
                    ax=ax_cm)
        ax_cm.set_title(f'{model_name_prefix} Confusion Matrix ({data_split_name})')
        ax_cm.set_xlabel('Predicted Label')
        ax_cm.set_ylabel('True Label')
        
        filename_data_split_part = data_split_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        cm_filename = f'{model_name_prefix.lower().replace(" ", "_")}_confusion_matrix_{filename_data_split_part}.png'
        cm_path = os.path.join(directory, cm_filename)
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close(fig_cm)
        print(f"{model_name_prefix} Confusion Matrix ({data_split_name}) saved to: {cm_path}")

    # 4. Model Coefficients / Weights & Biases Table - Conditional
    is_logistic_regression_like = False
    if len(model.layers) == 1 and isinstance(model.layers[0], Dense):
        weights_shape = model.layers[0].get_weights()[0].shape
        biases_shape = model.layers[0].get_weights()[1].shape
        if weights_shape[0] == len(feature_names) and weights_shape[1] == 7 and biases_shape[0] == 7:
            is_logistic_regression_like = True

    if is_logistic_regression_like:
        weights, biases = model.layers[0].get_weights()
        
        if isinstance(feature_names, str):
            feature_names_list = [feature_names]
        else:
            feature_names_list = list(feature_names)

        if weights.shape[0] == len(feature_names_list):
            coeffs_df = pd.DataFrame(weights, index=feature_names_list, columns=[f'To_Q{i+3}' for i in range(7)])
            coeffs_df.loc['Bias'] = biases
            coeffs_df = coeffs_df.applymap(lambda x: round(x, 4) if isinstance(x, (float, int)) else x)

            fig_betas, ax_betas = plt.subplots(figsize=(max(8, coeffs_df.shape[1]*1.5), max(4, len(coeffs_df) * 0.5))) 
            ax_betas.axis('tight'); ax_betas.axis('off')

            cell_text_betas = []
            for r_idx in coeffs_df.index:
                row_values = [f"{coeffs_df.loc[r_idx, c_idx]:.4f}" if isinstance(coeffs_df.loc[r_idx, c_idx], float) else str(coeffs_df.loc[r_idx, c_idx]) for c_idx in coeffs_df.columns]
                cell_text_betas.append(row_values)
            
            table_betas = ax_betas.table(cellText=cell_text_betas,
                                         rowLabels=coeffs_df.index,
                                         colLabels=coeffs_df.columns,
                                         cellLoc='center', loc='center', colWidths=[0.12]*len(coeffs_df.columns))
            table_betas.auto_set_font_size(False); table_betas.set_fontsize(9); table_betas.scale(1.1, 1.1)
            plt.title(f"{model_name_prefix} Model Coefficients", pad=20) 
            betas_path = os.path.join(directory, f'{model_name_prefix.lower().replace(" ", "_")}_model_coefficients_table.png') 
            plt.savefig(betas_path, bbox_inches='tight'); plt.close(fig_betas)
            print(f"Model coefficients table for {model_name_prefix} saved to: {betas_path}")
            print(f"\n{model_name_prefix} Model Coefficients:\n", coeffs_df)
        else:
            print(f"\nSkipping Model Coefficients table for {model_name_prefix}: Mismatch between feature names ({len(feature_names_list)}) and extracted weights features ({weights.shape[0]}).")

    elif len(model.layers) > 0 and all(isinstance(layer, Dense) for layer in model.layers) and model.layers[-1].units == 7:
        print(f"\nGenerating Multi-Layer Neural Network Weights and Biases tables for {model_name_prefix}...")
        
        num_keras_layers = len(model.layers)

        for i, layer in enumerate(model.layers):
            layer_weights, layer_biases = layer.get_weights()
            num_current_layer_neurons = layer.units 

            input_names_for_weights_table = []
            expected_input_dim_for_weights = 0

            if i == 0: 
                input_names_for_weights_table = feature_names
                expected_input_dim_for_weights = len(feature_names)
            else: 
                num_prev_layer_neurons = model.layers[i-1].units
                input_names_for_weights_table = [f'From L{i-1}_N{j+1}' for j in range(num_prev_layer_neurons)]
                expected_input_dim_for_weights = num_prev_layer_neurons
            
            if layer_weights.shape[0] != expected_input_dim_for_weights:
                print(f"Skipping weights for {model_name_prefix} Keras Layer {i}: Mismatch between expected input dimension ({expected_input_dim_for_weights}) and layer's input weights dimension ({layer_weights.shape[0]}).")
            else:
                if i == num_keras_layers - 1:
                    weight_table_columns = [f'To Q{k+3}' for k in range(num_current_layer_neurons)]
                else:
                    weight_table_columns = [f'To L{i}_N{k+1}' for k in range(num_current_layer_neurons)]
                
                weights_df = pd.DataFrame(layer_weights, index=input_names_for_weights_table, columns=weight_table_columns)

                fig_w, ax_w = plt.subplots(figsize=(max(6, num_current_layer_neurons*1.3), max(3, len(input_names_for_weights_table)*0.4 + 1.5)))
                ax_w.axis('tight'); ax_w.axis('off')
                cell_text_w = [[f"{val:.4f}" if isinstance(val, float) else str(val) for val in weights_df.loc[r]] for r in weights_df.index]
                
                table_w = ax_w.table(cellText=cell_text_w, rowLabels=weights_df.index, colLabels=weights_df.columns, cellLoc='center', loc='center')
                table_w.auto_set_font_size(False); table_w.set_fontsize(9); table_w.scale(1.1, 1.1)
                
                input_source_title = "InputFeat" if i == 0 else f"L{i-1}_Out"
                plt.title(f"{model_name_prefix} NN: {input_source_title} to L{i} Weights", pad=20)
                w_path = os.path.join(directory, f'{model_name_prefix.lower().replace(" ", "_")}_nn_{input_source_title.lower()}_to_l{i}_weights.png')
                plt.savefig(w_path, bbox_inches='tight'); plt.close(fig_w)
                print(f"{model_name_prefix} NN {input_source_title} to L{i} Weights:\n{weights_df.round(4)}")

            if i == num_keras_layers -1:
                bias_table_rows = [f'L{i}_Q{k+3}_Bias' for k in range(num_current_layer_neurons)]
            else:
                bias_table_rows = [f'L{i}_N{k+1} Bias' for k in range(num_current_layer_neurons)]

            if len(layer_biases) != num_current_layer_neurons:
                 print(f"Skipping biases for {model_name_prefix} Keras Layer {i}: Mismatch between number of neurons ({num_current_layer_neurons}) and number of biases ({len(layer_biases)}).")
            else:
                biases_df = pd.DataFrame(layer_biases, index=bias_table_rows, columns=['Bias Value'])
                
                fig_b, ax_b = plt.subplots(figsize=(6, max(3, num_current_layer_neurons*0.4 + 1.5)))
                ax_b.axis('tight'); ax_b.axis('off')
                cell_text_b = [[f"{val:.4f}" if isinstance(val, float) else str(val) for val in biases_df.loc[r]] for r in biases_df.index]

                table_b = ax_b.table(cellText=cell_text_b, rowLabels=biases_df.index, colLabels=biases_df.columns, cellLoc='center', loc='center')
                table_b.auto_set_font_size(False); table_b.set_fontsize(10); table_b.scale(1.1, 1.1)
                plt.title(f"{model_name_prefix} NN: Keras Layer {i} Biases", pad=20)
                b_path = os.path.join(directory, f'{model_name_prefix.lower().replace(" ", "_")}_nn_l{i}_biases.png')
                plt.savefig(b_path, bbox_inches='tight'); plt.close(fig_b)
                print(f"{model_name_prefix} NN Keras Layer {i} Biases table saved to: {b_path}")
                print(f"\n{model_name_prefix} NN Keras Layer {i} Biases:\n{biases_df.round(4)}")
            
    else:
        print(f"\nSkipping Model Coefficients/Weights table for {model_name_prefix}: Model structure not recognized as simple logistic regression or multi-layer Dense NN.")


def neural_network_neurons_sensibility_analysis(df, feature_cols, target_col, base_nn_regression_dir):
    """
    Performs a sensitivity analysis on the number of neurons in the hidden layer of a neural network,
    including a logistic regression equivalent (0 hidden neurons).
    Trains models with 0 to 10 effective hidden neurons, saves their graphs and metrics in separate folders,
    reports the configuration with the highest accuracy on the full dataset, and plots overall accuracy.
    If feature_cols > 2, decision boundaries are plotted in PCA space. Includes Early Stopping.
    Also saves PCA explained variance and loadings tables if PCA is used for plotting.
    """
    print("\n--- Starting Neural Network and Logistic Regression Neuron Sensibility Analysis ---")
    best_full_data_accuracy = -1.0
    best_config_description = "" 
    best_config_neuron_count_for_plot = -1

    neuron_config_counts_history = [] # For x-axis of the plot (0 for LR, 1-10 for NN neurons)
    full_data_accuracy_history = []   # To store corresponding accuracies

    # Common data preparation (once)
    X_full_raw_original_scale = df[feature_cols].values
    y_full_raw_original_values = df[target_col].values

    # Map quality 3-9 to 0-6
    y_full_mapped_0_6 = y_full_raw_original_values - 3
    
    # One-hot encode for Keras
    y_full_ohe = to_categorical(y_full_mapped_0_6, num_classes=7)

    # Use y_full_mapped_0_6 for stratification as it's 1D
    X_train_raw, X_test_raw, y_train_mapped_0_6, y_test_mapped_0_6 = train_test_split(
        X_full_raw_original_scale, y_full_mapped_0_6, test_size=0.2, random_state=42, stratify=y_full_mapped_0_6
    )
    
    # One-hot encode train and test y
    y_train_ohe = to_categorical(y_train_mapped_0_6, num_classes=7)
    y_test_ohe = to_categorical(y_test_mapped_0_6, num_classes=7)
    
    # scaler = StandardScaler() # REMOVED - Data is assumed pre-scaled
    # X_train_scaled = scaler.fit_transform(X_train_raw) # REMOVED
    # X_test_scaled = scaler.transform(X_test_raw) # REMOVED
    # X_full_scaled = scaler.transform(X_full_raw)  # REMOVED
    
    # Use the raw (pre-scaled) data directly
    X_train_for_model = X_train_raw
    X_test_for_model = X_test_raw
    X_full_for_model = X_full_raw_original_scale

    # Define EarlyStopping callback for main models
    early_stopping_main = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    
    pca_for_plotting = None # Initialize PCA object for plotting, if needed
    X_test_plot_data = X_test_for_model # Will be X_test_raw (pre-scaled)
    X_full_plot_data = X_full_for_model  # Will be X_full_raw_original_scale (pre-scaled)
    plot_feature_names = feature_cols

    if len(feature_cols) > 2:
        print("\nPreparing PCA for decision boundary visualization (fit on training data)...")
        pca_for_plotting = PCA(n_components=2, random_state=42)
        pca_for_plotting.fit(X_train_for_model) # Fit PCA on the (pre-scaled) training data
        X_test_plot_data = pca_for_plotting.transform(X_test_for_model)
        X_full_plot_data = pca_for_plotting.transform(X_full_for_model)
        plot_feature_names = ['Principal Component 1', 'Principal Component 2']

        # Generate and save PCA explained variance table for the visualization PCA
        explained_variance_vis = pca_for_plotting.explained_variance_ratio_
        explained_variance_vis_df_data = {
            'Principal Component': ['PC1', 'PC2'],
            'Variance Explained Ratio': [f"{var:.4f}" for var in explained_variance_vis]
        }
        explained_variance_vis_df = pd.DataFrame(explained_variance_vis_df_data)
        total_variance_vis = explained_variance_vis.sum()
        total_row_vis = pd.DataFrame({
            'Principal Component': ['Total Explained (PC1+PC2)'],
            'Variance Explained Ratio': [f"{total_variance_vis:.4f}"]
        })
        explained_variance_vis_df = pd.concat([explained_variance_vis_df, total_row_vis], ignore_index=True)

        fig_ev_vis, ax_ev_vis = plt.subplots(figsize=(6, max(2.5, explained_variance_vis_df.shape[0] * 0.5)))
        ax_ev_vis.axis('tight'); ax_ev_vis.axis('off')
        ev_table_vis = ax_ev_vis.table(cellText=explained_variance_vis_df.values,
                                   colLabels=explained_variance_vis_df.columns,
                                   cellLoc='center', loc='center', colWidths=[0.5, 0.5])
        ev_table_vis.auto_set_font_size(False); ev_table_vis.set_fontsize(10); ev_table_vis.scale(1.1, 1.1)
        plt.title("PCA Explained Variance (for Visualization)", pad=20, fontsize=12)
        explained_variance_vis_path = os.path.join(base_nn_regression_dir, 'pca_visualization_explained_variance.png')
        plt.savefig(explained_variance_vis_path, bbox_inches='tight')
        plt.close(fig_ev_vis)
        print(f"PCA (for visualization) explained variance table saved to: {explained_variance_vis_path}")
        print("\nPCA (for visualization) Explained Variance Ratio:\n", explained_variance_vis_df)

        # Generate and save PCA component loadings table for the visualization PCA
        loadings_vis = pca_for_plotting.components_
        loadings_vis_df = pd.DataFrame(loadings_vis.T, columns=['PC1 Loading', 'PC2 Loading'], index=feature_cols)
        
        fig_load_vis, ax_load_vis = plt.subplots(figsize=(8, max(4, len(feature_cols) * 0.4 + 1)))
        ax_load_vis.axis('tight'); ax_load_vis.axis('off')
        cell_text_loadings_vis = [[f"{val:.4f}" for val in loadings_vis_df.loc[r]] for r in loadings_vis_df.index]
        load_table_vis = ax_load_vis.table(cellText=cell_text_loadings_vis,
                                       rowLabels=loadings_vis_df.index,
                                       colLabels=loadings_vis_df.columns,
                                       cellLoc='center', loc='center', colWidths=[0.3, 0.3])
        load_table_vis.auto_set_font_size(False); load_table_vis.set_fontsize(10); load_table_vis.scale(1.1, 1.1)
        plt.title(f"PCA Component Loadings (for Visualization)\n({len(feature_cols)} Original Features)", pad=20, fontsize=12)
        loadings_vis_path = os.path.join(base_nn_regression_dir, 'pca_visualization_component_loadings.png')
        plt.savefig(loadings_vis_path, bbox_inches='tight')
        plt.close(fig_load_vis)
        print(f"PCA (for visualization) component loadings table saved to: {loadings_vis_path}")
        print("\nPCA (for visualization) Component Loadings:\n", loadings_vis_df.round(4))


    for num_hidden_neurons_config in range(0, 21): 
        current_model_dir_path = ""
        model = None
        model_name_prefix = ""

        if num_hidden_neurons_config == 0: # Logistic Regression case
            model_name_prefix = "Logistic Regression (0 Hidden Neurons)"
            current_model_dir_name = "logistic_regression_0_hidden_neurons"
            current_model_dir_path = os.path.join(base_nn_regression_dir, current_model_dir_name)
            
            model = Sequential([
                Dense(7, activation='softmax', input_shape=(X_train_for_model.shape[1],))
            ])
            neuron_config_counts_history.append(0)
        else: # Neural Network with hidden layer case
            model_name_prefix = f"NN {num_hidden_neurons_config} Hidden Neurons"
            current_model_dir_name = f"nn_{num_hidden_neurons_config}_hidden_neurons"
            current_model_dir_path = os.path.join(base_nn_regression_dir, current_model_dir_name)

            model = Sequential([
                Dense(num_hidden_neurons_config, activation='relu', input_shape=(X_train_for_model.shape[1],)),
                Dense(7, activation='softmax')
            ])
            neuron_config_counts_history.append(num_hidden_neurons_config)

        print(f"\n--- Training and Evaluating: {model_name_prefix} ---")
        os.makedirs(current_model_dir_path, exist_ok=True)
        print(f"Output directory for this model: {current_model_dir_path}")

        optimizer = Adam(learning_rate=0.01)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        print(f"Training {model_name_prefix}...")
        history = model.fit(X_train_for_model, y_train_ohe, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping_main])

        loss_test, accuracy_test = model.evaluate(X_test_for_model, y_test_ohe, verbose=0)
        print(f"\n{model_name_prefix} - Test Loss: {loss_test:.4f}")
        print(f"{model_name_prefix} - Test Accuracy: {accuracy_test:.4f}")

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name_prefix}: Model Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
        plt.ylim(bottom=0) # Correct for loss, as it can exceed 1

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{model_name_prefix}: Model Accuracy')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
        plt.ylim(0, 1) # Changed: Accuracy is between 0 and 1
        
        history_plot_filename = f'{model_name_prefix.lower().replace(" ", "_").replace("(", "").replace(")", "")}_training_history.png'
        plt.savefig(os.path.join(current_model_dir_path, history_plot_filename))
        plt.close()
        print(f"Training history plot saved to: {os.path.join(current_model_dir_path, history_plot_filename)}")

        # Decision Boundary Plotting Logic uses the main 'model'
        # X_test_plot_data and X_full_plot_data are either original scaled 2D features or PCA-transformed 2D features
        # plot_feature_names are corresponding names
        # pca_for_plotting is None if original features were 2D, or the fitted PCA object if >2 features
        print(f"\nPlotting decision boundaries for {model_name_prefix}...")
        plot_decision_boundary(X_test_plot_data, y_test_mapped_0_6, model, None, plot_feature_names, current_model_dir_path, "Test Set", model_name_prefix, pca_transformer=pca_for_plotting)
        plot_decision_boundary(X_full_plot_data, y_full_mapped_0_6, model, None, plot_feature_names, current_model_dir_path, "Full Data", model_name_prefix, pca_transformer=pca_for_plotting)
        
        # Model metrics are always based on the original model trained on original features
        plot_model_metrics(model, X_test_for_model, y_test_mapped_0_6, X_train_for_model, y_train_mapped_0_6, X_full_for_model, y_full_mapped_0_6, feature_cols, current_model_dir_path, model_name_prefix)

        loss_full, accuracy_full = model.evaluate(X_full_for_model, y_full_ohe, verbose=0)
        print(f"{model_name_prefix} - Full Data Accuracy: {accuracy_full:.4f}")
        
        full_data_accuracy_history.append(accuracy_full)

        if accuracy_full > best_full_data_accuracy:
            best_full_data_accuracy = accuracy_full
            best_config_description = model_name_prefix
            best_config_neuron_count_for_plot = num_hidden_neurons_config
            print(f"New best full data accuracy: {best_full_data_accuracy:.4f} with {best_config_description}.")

    print("\n--- Neural Network and Logistic Regression Sensibility Analysis Complete ---")
    if best_config_description:
        print(f"The highest accuracy on the Full Data ({best_full_data_accuracy:.4f}) was achieved with: {best_config_description}.")
    else:
        print("Could not determine the best configuration (no models improved accuracy).")
        
    if neuron_config_counts_history and full_data_accuracy_history:
        plt.figure(figsize=(10, 6))
        plt.plot(neuron_config_counts_history, full_data_accuracy_history, marker='o', linestyle='-', label='Full Data Accuracy')
        if best_config_description: # Check if a best model was found
            plt.plot(best_config_neuron_count_for_plot, best_full_data_accuracy, marker='*', markersize=15, color='red', label=f'Best: {best_config_description} ({best_full_data_accuracy:.4f})')
        
        plt.title('Model Configuration vs. Full Data Accuracy')
        plt.xlabel('Number of Hidden Neurons (0 for Logistic Regression)')
        plt.ylabel('Accuracy on Full Data')
        plt.xticks(neuron_config_counts_history)
        plt.legend()
        plt.grid(True)
        summary_plot_path = os.path.join(base_nn_regression_dir, 'model_config_accuracy_summary.png')
        plt.savefig(summary_plot_path)
        plt.close()
        print(f"Accuracy summary plot saved to: {summary_plot_path}")

    return best_config_description, best_full_data_accuracy


def neural_network_two_hidden_layers_sensibility_analysis(df, feature_cols, target_col, base_nn_2_layers_dir):
    """
    Performs a sensitivity analysis on the number of neurons in two hidden layers of a neural network.
    Trains models with 1 to 20 neurons in each of the two hidden layers (400 models total).
    Saves graphs, metrics, weights, and biases for each model in separate folders.
    Reports the configuration with the highest accuracy on the full dataset.
    Plots a line graph and a heatmap of all model accuracies on the full dataset.
    If feature_cols > 2, decision boundaries are plotted in PCA space. Includes Early Stopping.
    Also saves PCA explained variance and loadings tables if PCA is used for plotting.
    """
    num_neuron_options = 20 # Moved to the top
    print(f"\n--- Starting Neural Network Two Hidden Layers Sensibility Analysis (1-{num_neuron_options} Neurons per Layer) ---")
    
    all_configs_accuracies_data = [] 
    best_full_data_accuracy = -1.0
    best_config_description = ""
    best_config_neurons = (0,0)

    X_full_raw_original_scale = df[feature_cols].values
    y_full_raw_original_values = df[target_col].values

    y_full_mapped_0_6 = y_full_raw_original_values - 3
    y_full_ohe = to_categorical(y_full_mapped_0_6, num_classes=7)
    
    X_train_raw, X_test_raw, y_train_mapped_0_6, y_test_mapped_0_6 = train_test_split(
        X_full_raw_original_scale, y_full_mapped_0_6, test_size=0.2, random_state=42, stratify=y_full_mapped_0_6
    )
    y_train_ohe = to_categorical(y_train_mapped_0_6, num_classes=7)
    y_test_ohe = to_categorical(y_test_mapped_0_6, num_classes=7)
    
    X_train_for_model = X_train_raw
    X_test_for_model = X_test_raw
    X_full_for_model = X_full_raw_original_scale

    accuracies_matrix = np.zeros((num_neuron_options, num_neuron_options)) 

    # Define EarlyStopping callback for main models
    early_stopping_main = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

    pca_for_plotting = None # Initialize PCA object for plotting, if needed
    X_test_plot_data = X_test_for_model
    X_full_plot_data = X_full_for_model
    plot_feature_names = feature_cols

    if len(feature_cols) > 2:
        print("\nPreparing PCA for decision boundary visualization (fit on training data)...")
        pca_for_plotting = PCA(n_components=2, random_state=42)
        pca_for_plotting.fit(X_train_for_model) # Fit PCA on the (pre-scaled) training data
        X_test_plot_data = pca_for_plotting.transform(X_test_for_model)
        X_full_plot_data = pca_for_plotting.transform(X_full_for_model)
        plot_feature_names = ['Principal Component 1', 'Principal Component 2']

        # Generate and save PCA explained variance table for the visualization PCA
        explained_variance_vis = pca_for_plotting.explained_variance_ratio_
        explained_variance_vis_df_data = {
            'Principal Component': ['PC1', 'PC2'],
            'Variance Explained Ratio': [f"{var:.4f}" for var in explained_variance_vis]
        }
        explained_variance_vis_df = pd.DataFrame(explained_variance_vis_df_data)
        total_variance_vis = explained_variance_vis.sum()
        total_row_vis = pd.DataFrame({
            'Principal Component': ['Total Explained (PC1+PC2)'],
            'Variance Explained Ratio': [f"{total_variance_vis:.4f}"]
        })
        explained_variance_vis_df = pd.concat([explained_variance_vis_df, total_row_vis], ignore_index=True)

        fig_ev_vis, ax_ev_vis = plt.subplots(figsize=(6, max(2.5, explained_variance_vis_df.shape[0] * 0.5)))
        ax_ev_vis.axis('tight'); ax_ev_vis.axis('off')
        ev_table_vis = ax_ev_vis.table(cellText=explained_variance_vis_df.values,
                                   colLabels=explained_variance_vis_df.columns,
                                   cellLoc='center', loc='center', colWidths=[0.5, 0.5])
        ev_table_vis.auto_set_font_size(False); ev_table_vis.set_fontsize(10); ev_table_vis.scale(1.1, 1.1)
        plt.title("PCA Explained Variance (for Visualization)", pad=20, fontsize=12)
        explained_variance_vis_path = os.path.join(base_nn_2_layers_dir, 'pca_visualization_explained_variance.png')
        plt.savefig(explained_variance_vis_path, bbox_inches='tight')
        plt.close(fig_ev_vis)
        print(f"PCA (for visualization) explained variance table saved to: {explained_variance_vis_path}")
        print("\nPCA (for visualization) Explained Variance Ratio:\n", explained_variance_vis_df)

        # Generate and save PCA component loadings table for the visualization PCA
        loadings_vis = pca_for_plotting.components_
        loadings_vis_df = pd.DataFrame(loadings_vis.T, columns=['PC1 Loading', 'PC2 Loading'], index=feature_cols)
        
        fig_load_vis, ax_load_vis = plt.subplots(figsize=(8, max(4, len(feature_cols) * 0.4 + 1)))
        ax_load_vis.axis('tight'); ax_load_vis.axis('off')
        cell_text_loadings_vis = [[f"{val:.4f}" for val in loadings_vis_df.loc[r]] for r in loadings_vis_df.index]
        load_table_vis = ax_load_vis.table(cellText=cell_text_loadings_vis,
                                       rowLabels=loadings_vis_df.index,
                                       colLabels=loadings_vis_df.columns,
                                       cellLoc='center', loc='center', colWidths=[0.3, 0.3])
        load_table_vis.auto_set_font_size(False); load_table_vis.set_fontsize(10); load_table_vis.scale(1.1, 1.1)
        plt.title(f"PCA Component Loadings (for Visualization)\n({len(feature_cols)} Original Features)", pad=20, fontsize=12)
        loadings_vis_path = os.path.join(base_nn_2_layers_dir, 'pca_visualization_component_loadings.png')
        plt.savefig(loadings_vis_path, bbox_inches='tight')
        plt.close(fig_load_vis)
        print(f"PCA (for visualization) component loadings table saved to: {loadings_vis_path}")
        print("\nPCA (for visualization) Component Loadings:\n", loadings_vis_df.round(4))


    for neurons_l1 in range(1, num_neuron_options + 1):
        for neurons_l2 in range(1, num_neuron_options + 1):
            model_name_prefix = f"NN_L1_{neurons_l1}_L2_{neurons_l2}" 
            current_model_dir_name = f"nn_L1_{neurons_l1}_L2_{neurons_l2}_neurons"
            current_model_dir_path = os.path.join(base_nn_2_layers_dir, current_model_dir_name)
            os.makedirs(current_model_dir_path, exist_ok=True)
            
            full_model_description = f"NN with {neurons_l1} Neurons in Layer 1 and {neurons_l2} Neurons in Layer 2"
            print(f"\n--- Training and Evaluating: {full_model_description} ---")
            print(f"Output directory for this model: {current_model_dir_path}")

            # Main model trained on original scaled features
            model = Sequential([
                Dense(neurons_l1, activation='relu', input_shape=(X_train_for_model.shape[1],)),
                Dense(neurons_l2, activation='relu'),
                Dense(7, activation='softmax')
            ])

            optimizer = Adam(learning_rate=0.01)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

            print(f"Training {full_model_description}...")
            history = model.fit(X_train_for_model, y_train_ohe, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping_main])

            loss_test, accuracy_test = model.evaluate(X_test_for_model, y_test_ohe, verbose=0)
            print(f"\n{full_model_description} - Test Loss: {loss_test:.4f}, Test Accuracy: {accuracy_test:.4f}")

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'{full_model_description}:\nModel Loss', fontsize=10)
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
            plt.ylim(bottom=0) # Correct for loss, as it can exceed 1

            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title(f'{full_model_description}:\nModel Accuracy', fontsize=10)
            plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
            plt.ylim(0, 1) # Changed: Accuracy is between 0 and 1
            
            plt.tight_layout(rect=[0, 0, 1, 0.96]) 
            history_plot_filename = f'{model_name_prefix.lower().replace(" ", "_")}_training_history.png'
            plt.savefig(os.path.join(current_model_dir_path, history_plot_filename))
            plt.close()
            print(f"Training history plot saved to: {os.path.join(current_model_dir_path, history_plot_filename)}")

            # Decision Boundary Plotting Logic uses the main 'model'
            print(f"\nPlotting decision boundaries for {full_model_description}...")
            plot_decision_boundary(X_test_plot_data, y_test_mapped_0_6, model, None, plot_feature_names, current_model_dir_path, "Test Set", full_model_description, pca_transformer=pca_for_plotting)
            plot_decision_boundary(X_full_plot_data, y_full_mapped_0_6, model, None, plot_feature_names, current_model_dir_path, "Full Data", full_model_description, pca_transformer=pca_for_plotting)
                        
            # Model metrics are always based on the original model trained on original features
            plot_model_metrics(model, X_test_for_model, y_test_mapped_0_6, X_train_for_model, y_train_mapped_0_6, X_full_for_model, y_full_mapped_0_6, feature_cols, current_model_dir_path, full_model_description)

            loss_full, accuracy_full = model.evaluate(X_full_for_model, y_full_ohe, verbose=0)
            print(f"{full_model_description} - Full Data Accuracy: {accuracy_full:.4f}")
            
            all_configs_accuracies_data.append(((neurons_l1, neurons_l2), accuracy_full, model_name_prefix))
            accuracies_matrix[neurons_l1-1, neurons_l2-1] = accuracy_full 

            if accuracy_full > best_full_data_accuracy:
                best_full_data_accuracy = accuracy_full
                best_config_description = full_model_description
                best_config_neurons = (neurons_l1, neurons_l2)
                print(f"New best full data accuracy: {best_full_data_accuracy:.4f} with {best_config_description}.")

    print("\n--- Neural Network Two Hidden Layers Sensibility Analysis Complete (1-20 Neurons) ---")
    if best_config_description:
        print(f"The highest accuracy on the Full Data ({best_full_data_accuracy:.4f}) was achieved with: {best_config_description} (L1: {best_config_neurons[0]}, L2: {best_config_neurons[1]}).")
    else:
        print("Could not determine the best configuration (no models improved accuracy).")

    if all_configs_accuracies_data:
        all_configs_accuracies_data.sort(key=lambda x: (x[0][0], x[0][1]))
        
        config_labels = [f"L1:{cfg[0][0]},L2:{cfg[0][1]}" for cfg in all_configs_accuracies_data]
        accuracies = [cfg[1] for cfg in all_configs_accuracies_data]
        
        plt.figure(figsize=(max(25, len(config_labels) * 0.075), 10)) 
        plt.plot(config_labels, accuracies, marker='o', linestyle='-', label='Full Data Accuracy', markersize=3, linewidth=0.8) 
        
        if best_config_description: 
            best_config_label = f"L1:{best_config_neurons[0]},L2:{best_config_neurons[1]}"
            try:
                best_config_idx = config_labels.index(best_config_label)
                plt.plot(best_config_idx, best_full_data_accuracy, marker='*', markersize=12, color='red', 
                         label=f'Best: {best_config_label} ({best_full_data_accuracy:.4f})')
            except ValueError:
                print(f"Warning: Best configuration label '{best_config_label}' not found in config_labels for plotting.")

        plt.xlabel("Model Configuration (Neurons L1, Neurons L2)")
        plt.ylabel("Accuracy on Full Data")
        plt.title(f"Accuracy of Two Hidden Layer NN Configurations (1-{num_neuron_options} Neurons/Layer) (Full Data)")
        plt.xticks(rotation=90, ha='right', fontsize=4) 
        
        min_acc_val = min(accuracies) if accuracies else 0.0
        max_acc_val = max(accuracies) if accuracies else 1.0
        y_tick_step = 0.05 if (max_acc_val - min_acc_val) > 0.2 else 0.01
        y_ticks_start = np.floor(min_acc_val / y_tick_step) * y_tick_step if accuracies else 0.0
        y_ticks_end = np.ceil(max_acc_val / y_tick_step) * y_tick_step + y_tick_step if accuracies else 1.0
        if y_ticks_start > min_acc_val and accuracies : y_ticks_start -= y_tick_step
        
        plt.yticks(np.arange(y_ticks_start, y_ticks_end, y_tick_step))
        plt.legend(fontsize=8) 
        plt.grid(axis='both', linestyle='--') 
        plt.tight_layout()
        
        line_plot_path = os.path.join(base_nn_2_layers_dir, f'all_models_accuracy_line_plot_1_to_{num_neuron_options}_neurons.png')
        plt.savefig(line_plot_path)
        plt.close()
        print(f"Line plot of all model accuracies saved to: {line_plot_path}")

    if np.any(accuracies_matrix): 
        plt.figure(figsize=(14, 12)) 
        neuron_counts_axis = list(range(1, num_neuron_options + 1))
        sns.heatmap(accuracies_matrix, annot=True, fmt=".3f", cmap="viridis",
                    xticklabels=neuron_counts_axis, yticklabels=neuron_counts_axis,
                    cbar_kws={'label': 'Accuracy on Full Data'})
        plt.xlabel("Number of Neurons in Hidden Layer 2")
        plt.ylabel("Number of Neurons in Hidden Layer 1")
        plt.title(f"Heatmap of Model Accuracy (Full Data)\nvs. Neuron Configuration (1-{num_neuron_options} Neurons/Layer)")
        heatmap_path = os.path.join(base_nn_2_layers_dir, f'accuracy_heatmap_L1_vs_L2_1_to_{num_neuron_options}_neurons.png')
        plt.savefig(heatmap_path, bbox_inches='tight')
        plt.close()
        print(f"Accuracy heatmap saved to: {heatmap_path}")
        
    return best_config_description, best_full_data_accuracy, best_config_neurons

#directorio para guardar los graficos
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, 'graficos', f'modelo_calidad_rojo_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}') # Changed here
os.makedirs(parent_dir, exist_ok=True)


#leer los datos
# For the red wine model, we only need red_wine_data
red_wine_data_path = os.path.join(current_dir, 'generate_color_csv', 'red_wine_predictions.csv')

try:
    red_wine_data = pd.read_csv(red_wine_data_path, sep=',')
    print(f"Successfully loaded red wine data from: {red_wine_data_path}")
    print("\nRed Wine Data Head:")
    print(red_wine_data.head())
    red_wine_data.name = 'red_wine_data_scaled' # Giving a more descriptive name
except FileNotFoundError:
    print(f"Error: Could not find red wine data at {red_wine_data_path}. Please ensure the file exists.")
    print("Cannot proceed with Red Wine Model without data.")
    red_wine_data = pd.DataFrame() # Assign empty DataFrame


if not red_wine_data.empty:
    #MODELO CALIDAD PARA VINO ROJO
    #directorio modelo vino rojo
    red_wine_dir = os.path.join(parent_dir, 'modelo_rojo') # This is the main folder for this script's outputs
    os.makedirs(red_wine_dir, exist_ok=True)

    #directorio modelo simple
    simple_dir = os.path.join(red_wine_dir, 'modelo_simple')
    os.makedirs(simple_dir, exist_ok=True)

    simple_nn_base_regression_dir = os.path.join(simple_dir, 'neural_network_and_lr_analysis_1_layer')
    simple_nn_dir_2_layers = os.path.join(simple_dir, 'neural_network_2_layers')
    os.makedirs(simple_nn_base_regression_dir, exist_ok=True)
    os.makedirs(simple_nn_dir_2_layers, exist_ok=True)

    #directorio modelo intermedio
    intermediate_dir = os.path.join(red_wine_dir, 'modelo_intermedio')
    os.makedirs(intermediate_dir, exist_ok=True)

    intermediate_nn_base_regression_dir = os.path.join(intermediate_dir, 'neural_network_and_lr_analysis_1_layer')
    intermediate_nn_dir_2_layers = os.path.join(intermediate_dir, 'neural_network_2_layers')
    os.makedirs(intermediate_nn_base_regression_dir, exist_ok=True)
    os.makedirs(intermediate_nn_dir_2_layers, exist_ok=True)


    #directorio modelo completo
    complete_dir = os.path.join(red_wine_dir, 'modelo_completo')
    os.makedirs(complete_dir, exist_ok=True)

    complete_nn_base_regression_dir = os.path.join(complete_dir, 'neural_network_and_lr_analysis_1_layer')
    complete_nn_dir_2_layers = os.path.join(complete_dir, 'neural_network_2_layers')
    os.makedirs(complete_nn_base_regression_dir, exist_ok=True)
    os.makedirs(complete_nn_dir_2_layers, exist_ok=True)


    #modelo simple 2 variables
    # Using features as per original script for red wine simple model
    feature_columns_simple_red = ['volatile acidity', 'total sulfur dioxide']
    target_column_red = 'quality'

    if all(col in red_wine_data.columns for col in feature_columns_simple_red) and target_column_red in red_wine_data.columns:
        scatterplot_vs_quality(red_wine_data, feature_columns_simple_red[0], feature_columns_simple_red[1], simple_dir)

        print("\n--- Red Wine Simple model: Combined Neural Network and Logistic Regression Sensibility Analysis (1 Hidden Layer) ---")
        neural_network_neurons_sensibility_analysis(red_wine_data, feature_columns_simple_red, target_column_red, simple_nn_base_regression_dir)

        print("\n--- Red Wine Simple model: Neural Network Sensibility Analysis (2 Hidden Layers) ---")
        neural_network_two_hidden_layers_sensibility_analysis(red_wine_data, feature_columns_simple_red, target_column_red, simple_nn_dir_2_layers)
    else:
        print(f"Skipping Red Wine Simple Model: Not all required columns ({feature_columns_simple_red + [target_column_red]}) found in red_wine_data.")

    #modelo intermedio 5 variables
    # Using features as per original script for red wine intermediate model
    feature_columns_intermediate_red = ['volatile acidity', 'total sulfur dioxide', 'alcohol', 'sulphates', 'citric acid']
    # target_column_red is already defined

    if all(col in red_wine_data.columns for col in feature_columns_intermediate_red) and target_column_red in red_wine_data.columns:
        plot_pca(red_wine_data, feature_columns_intermediate_red, target_column_red, intermediate_dir)

        print("\n--- Red Wine Intermediate model: Combined Neural Network and Logistic Regression Sensibility Analysis (1 Hidden Layer) ---")
        neural_network_neurons_sensibility_analysis(red_wine_data, feature_columns_intermediate_red, target_column_red, intermediate_nn_base_regression_dir)

        print("\n--- Red Wine Intermediate model: Neural Network Sensibility Analysis (2 Hidden Layers) ---")
        neural_network_two_hidden_layers_sensibility_analysis(red_wine_data, feature_columns_intermediate_red, target_column_red, intermediate_nn_dir_2_layers)
    else:
        print(f"Skipping Red Wine Intermediate Model: Not all required columns ({feature_columns_intermediate_red + [target_column_red]}) found in red_wine_data.")

    #modelo completo todas variables
    feature_columns_complete_red = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    # target_column_red is already defined

    if all(col in red_wine_data.columns for col in feature_columns_complete_red) and target_column_red in red_wine_data.columns:
        plot_pca(red_wine_data, feature_columns_complete_red, target_column_red, complete_dir)

        print("\n--- Red Wine Complete model: Combined Neural Network and Logistic Regression Sensibility Analysis (1 Hidden Layer) ---")
        neural_network_neurons_sensibility_analysis(red_wine_data, feature_columns_complete_red, target_column_red, complete_nn_base_regression_dir)

        print("\n--- Red Wine Complete model: Neural Network Sensibility Analysis (2 Hidden Layers) ---")
        neural_network_two_hidden_layers_sensibility_analysis(red_wine_data, feature_columns_complete_red, target_column_red, complete_nn_dir_2_layers)
    else:
        print(f"Skipping Red Wine Complete Model: Not all required columns ({feature_columns_complete_red + [target_column_red]}) found in red_wine_data.")
else:
    print("Skipping Red Wine Model processing as red_wine_data is empty.")


plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figures...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all') 