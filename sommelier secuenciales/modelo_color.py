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

    # Standardize features before PCA
    scaler_pca = StandardScaler()
    X_pca_scaled = scaler_pca.fit_transform(X_pca_raw)

    # 2. Apply PCA
    pca = PCA(n_components=2, random_state=42) # random_state for reproducibility
    X_pca_transformed = pca.fit_transform(X_pca_scaled)

    pca_plot_df = pd.DataFrame(data=X_pca_transformed, columns=['Principal Component 1', 'Principal Component 2'])
    # Add target for coloring, ensure it's treated categorically for consistent legend/palette
    # We'll use a temporary string column for hue to ensure discrete colors and legend entries
    temp_hue_col = target_col + "_str_hue"
    pca_plot_df[temp_hue_col] = df[target_col].astype(str).values # Ensure correct alignment if df index is not 0-based

    # 3. Create Scatter Plot
    plt.figure(figsize=(10, 7))

    unique_hue_values = sorted(pca_plot_df[temp_hue_col].unique())
    palette_map = {}
    legend_labels_map = {}

    # Specific for wine type, assuming '0' and '1' are the string values after astype(str)
    if '0' in unique_hue_values and '1' in unique_hue_values and len(unique_hue_values) == 2:
        palette_map = {'0': 'red', '1': 'yellow'}
        legend_labels_map = {'0': 'Red Wine (0)', '1': 'White Wine (1)'}
    else: # Fallback for generic cases
        colors = sns.color_palette(n_colors=len(unique_hue_values))
        for i, val_str in enumerate(unique_hue_values):
            palette_map[val_str] = colors[i]
            legend_labels_map[val_str] = val_str # Original string value as label

    scatter_plot_ax = sns.scatterplot(
        x='Principal Component 1',
        y='Principal Component 2',
        hue=temp_hue_col,
        data=pca_plot_df,
        palette=palette_map,
        alpha=0.7,
        edgecolor='k', # Add edgecolors for better visibility of points
        linewidth=0.5
    )
    plt.title(f'PCA of Wine Data ({len(feature_cols)} Features) by Type')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    handles, current_labels_str = scatter_plot_ax.get_legend_handles_labels()
    new_plot_labels = [legend_labels_map.get(lbl_str, lbl_str) for lbl_str in current_labels_str]
    plt.legend(handles, new_plot_labels, title='Wine Type')

    scatter_plot_path = os.path.join(directory, 'pca_scatter_plot.png')
    plt.savefig(scatter_plot_path, bbox_inches='tight')
    plt.close()
    print(f"PCA scatter plot saved to: {scatter_plot_path}")

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


def scatterplot_vs_type(df, x_col, y_col, output_dir):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue='type', palette=['red', 'yellow'])
    plt.title(f'Scatter Plot of {x_col} vs {y_col} by Wine Type')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.savefig(os.path.join(output_dir, f'scatterplot_vs_type_{x_col}_vs_{y_col}.png'))
    plt.close()

def plot_decision_boundary(X_plot, y_plot, model_to_predict_with, scaler_original, feature_names_plot, directory, data_split_name, model_name_prefix, pca_transformer=None):
    plt.figure(figsize=(10, 6))

    # X_plot are the 2D points to be scattered (either original 2D features or 2D PCA-transformed features)
    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # grid_points_2d are the coordinates in the 2D plot space
    grid_points_2d = np.c_[xx.ravel(), yy.ravel()]
    
    grid_points_for_model_prediction = grid_points_2d
    if pca_transformer:
        # If PCA was used for visualization, inverse_transform the 2D grid points
        # back to the original high-dimensional space before prediction.
        print(f"Inverse transforming {grid_points_2d.shape[0]} grid points using PCA for model prediction...")
        grid_points_for_model_prediction = pca_transformer.inverse_transform(grid_points_2d)

    # Predict using the main model (model_to_predict_with)
    Z = model_to_predict_with.predict(grid_points_for_model_prediction)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu) 
    # Scatter the provided X_plot points (these are already in 2D)
    scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, cmap=plt.cm.RdYlBu, edgecolors='k') 
    plt.xlabel(feature_names_plot[0])
    plt.ylabel(feature_names_plot[1])
    
    title = f'{model_name_prefix}: Decision Boundary ({data_split_name}) - {feature_names_plot[0]} vs {feature_names_plot[1]}'
    plt.title(title)
    
    handles, labels = scatter.legend_elements()
    legend_labels = ['Red Wine (0)', 'White Wine (1)']
    if len(handles) == 1 and len(np.unique(y_plot)) == 1:
        if np.unique(y_plot)[0] == 0:
            legend_labels = ['Red Wine (0)']
        else:
            legend_labels = ['White Wine (1)']
    plt.legend(handles, legend_labels, title="Wine Type")
    
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
    y_pred_test_for_report = (y_pred_proba_test_for_report > 0.5).astype(int).flatten()

    # 2. Classification Metrics Table (based on Test Set)
    report_dict_test = classification_report(y_test, y_pred_test_for_report, target_names=['Red Wine (0)', 'White Wine (1)'], output_dict=True)
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
    print(f"\nClassification Report for {model_name_prefix} (Test Set):\n", classification_report(y_test, y_pred_test_for_report, target_names=['Red Wine (0)', 'White Wine (1)']))

    # --- Full Data Set Metrics ---
    print(f"\n--- {model_name_prefix} Metrics on Full Data Set ---")
    y_pred_proba_full = model.predict(X_full_scaled)
    y_pred_full_for_report = (y_pred_proba_full > 0.5).astype(int).flatten()

    report_dict_full = classification_report(y_full, y_pred_full_for_report, target_names=['Red Wine (0)', 'White Wine (1)'], output_dict=True)
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
    print(f"\nClassification Report for {model_name_prefix} (Full Data):\n", classification_report(y_full, y_pred_full_for_report, target_names=['Red Wine (0)', 'White Wine (1)']))

    # 3. Confusion Matrices (Test Set and Full Data)
    print(f"\n--- {model_name_prefix} Confusion Matrices ---")
    datasets_for_cm = {
        "Test Set": (X_test_scaled, y_test),
        "Full Data": (X_full_scaled, y_full)
    }

    for data_split_name, (X_data_scaled, y_data_true) in datasets_for_cm.items():
        y_pred_proba = model.predict(X_data_scaled)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        cm = confusion_matrix(y_data_true, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Red Wine (0)', 'White Wine (1)'], 
                    yticklabels=['Red Wine (0)', 'White Wine (1)'],
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
        if weights_shape[0] == len(feature_names) and weights_shape[1] == 1 and biases_shape[0] == 1:
            is_logistic_regression_like = True

    if is_logistic_regression_like:
        weights, biases = model.layers[0].get_weights() 
        betas = weights.flatten() 
        bias = biases[0] 

        if isinstance(feature_names, str):
            feature_names_list = [feature_names]
        else:
            feature_names_list = list(feature_names)

        if len(betas) == len(feature_names_list):
            coeffs_data = {'Feature': feature_names_list + ['Bias'],
                           'Coefficient': list(betas) + [bias]}
            betas_df = pd.DataFrame(coeffs_data)
            betas_df['Coefficient'] = betas_df['Coefficient'].apply(lambda x: round(x, 4) if isinstance(x, (float,int)) else x)

            fig_betas, ax_betas = plt.subplots(figsize=(6, max(3, len(betas_df) * 0.5))) 
            ax_betas.axis('tight')
            ax_betas.axis('off')

            cell_text_betas = []
            for r_idx in betas_df.index:
                feature_val = betas_df.loc[r_idx, 'Feature']
                coeff_val = betas_df.loc[r_idx, 'Coefficient']
                
                if isinstance(coeff_val, float):
                    formatted_coeff = "{:.4f}".format(coeff_val)
                else: 
                    formatted_coeff = str(coeff_val)
                cell_text_betas.append([str(feature_val), formatted_coeff])

            table_betas = ax_betas.table(cellText=cell_text_betas,
                                         colLabels=betas_df.columns,
                                         cellLoc='center', loc='center', colWidths=[0.4, 0.4])
            table_betas.auto_set_font_size(False)
            table_betas.set_fontsize(10)
            table_betas.scale(1.2, 1.2)
            plt.title(f"{model_name_prefix} Model Coefficients (Betas)", pad=20) 
            betas_path = os.path.join(directory, f'{model_name_prefix.lower().replace(" ", "_")}_model_coefficients_table.png') 
            plt.savefig(betas_path, bbox_inches='tight')
            plt.close(fig_betas)
            print(f"Model coefficients table for {model_name_prefix} saved to: {betas_path}")
            print(f"\n{model_name_prefix} Model Coefficients (Betas):\n", betas_df)
        else:
            print(f"\nSkipping Model Coefficients table for {model_name_prefix}: Mismatch between feature names ({len(feature_names_list)}) and extracted betas ({len(betas)}).")

    elif len(model.layers) > 1 and all(isinstance(layer, Dense) for layer in model.layers) and model.layers[-1].units == 1:
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
                print(f"{model_name_prefix} NN Keras Layer {i} Weights table saved to: {w_path}")
                print(f"\n{model_name_prefix} NN {input_source_title} to L{i} Weights:\n{weights_df.round(4)}")

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
    X_full_raw = df[feature_cols].values
    y_full_raw = df[target_col].values
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_full_raw, y_full_raw, test_size=0.2, random_state=42, stratify=y_full_raw)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    X_full_scaled = scaler.transform(X_full_raw)

    # Define EarlyStopping callback for main models
    early_stopping_main = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    
    pca_for_plotting = None # Initialize PCA object for plotting, if needed
    X_test_plot_data = X_test_scaled
    X_full_plot_data = X_full_scaled
    plot_feature_names = feature_cols

    if len(feature_cols) > 2:
        print("\nPreparing PCA for decision boundary visualization (fit on training data)...")
        pca_for_plotting = PCA(n_components=2, random_state=42)
        pca_for_plotting.fit(X_train_scaled) # Fit PCA on the scaled training data
        X_test_plot_data = pca_for_plotting.transform(X_test_scaled)
        X_full_plot_data = pca_for_plotting.transform(X_full_scaled)
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


    for num_hidden_neurons_config in range(0, 11): 
        current_model_dir_path = ""
        model = None
        model_name_prefix = ""

        if num_hidden_neurons_config == 0: # Logistic Regression case
            model_name_prefix = "Logistic Regression (0 Hidden Neurons)"
            current_model_dir_name = "logistic_regression_0_hidden_neurons"
            current_model_dir_path = os.path.join(base_nn_regression_dir, current_model_dir_name)
            
            model = Sequential([
                Dense(1, activation='sigmoid', input_shape=(X_train_scaled.shape[1],))
            ])
            neuron_config_counts_history.append(0)
        else: # Neural Network with hidden layer case
            model_name_prefix = f"NN {num_hidden_neurons_config} Hidden Neurons"
            current_model_dir_name = f"nn_{num_hidden_neurons_config}_hidden_neurons"
            current_model_dir_path = os.path.join(base_nn_regression_dir, current_model_dir_name)

            model = Sequential([
                Dense(num_hidden_neurons_config, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                Dense(1, activation='sigmoid')
            ])
            neuron_config_counts_history.append(num_hidden_neurons_config)

        print(f"\n--- Training and Evaluating: {model_name_prefix} ---")
        os.makedirs(current_model_dir_path, exist_ok=True)
        print(f"Output directory for this model: {current_model_dir_path}")

        optimizer = Adam(learning_rate=0.01)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        print(f"Training {model_name_prefix}...")
        history = model.fit(X_train_scaled, y_train_raw, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping_main])

        loss_test, accuracy_test = model.evaluate(X_test_scaled, y_test_raw, verbose=0)
        print(f"\n{model_name_prefix} - Test Loss: {loss_test:.4f}")
        print(f"{model_name_prefix} - Test Accuracy: {accuracy_test:.4f}")

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name_prefix}: Model Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
        plt.ylim(0, 1) # Set y-axis limits for Loss plot

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{model_name_prefix}: Model Accuracy')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
        plt.ylim(0, 1) # Set y-axis limits for Accuracy plot
        
        history_plot_filename = f'{model_name_prefix.lower().replace(" ", "_").replace("(", "").replace(")", "")}_training_history.png'
        plt.savefig(os.path.join(current_model_dir_path, history_plot_filename))
        plt.close()
        print(f"Training history plot saved to: {os.path.join(current_model_dir_path, history_plot_filename)}")

        # Decision Boundary Plotting Logic uses the main 'model'
        # X_test_plot_data and X_full_plot_data are either original scaled 2D features or PCA-transformed 2D features
        # plot_feature_names are corresponding names
        # pca_for_plotting is None if original features were 2D, or the fitted PCA object if >2 features
        print(f"\nPlotting decision boundaries for {model_name_prefix}...")
        plot_decision_boundary(X_test_plot_data, y_test_raw, model, scaler, plot_feature_names, current_model_dir_path, "Test Set", model_name_prefix, pca_transformer=pca_for_plotting)
        plot_decision_boundary(X_full_plot_data, y_full_raw, model, scaler, plot_feature_names, current_model_dir_path, "Full Data", model_name_prefix, pca_transformer=pca_for_plotting)
        
        # Model metrics are always based on the original model trained on original features
        plot_model_metrics(model, X_test_scaled, y_test_raw, X_train_scaled, y_train_raw, X_full_scaled, y_full_raw, feature_cols, current_model_dir_path, model_name_prefix)

        loss_full, accuracy_full = model.evaluate(X_full_scaled, y_full_raw, verbose=0)
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

    X_full_raw = df[feature_cols].values
    y_full_raw = df[target_col].values
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_full_raw, y_full_raw, test_size=0.2, random_state=42, stratify=y_full_raw
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    X_full_scaled = scaler.transform(X_full_raw) 

    accuracies_matrix = np.zeros((num_neuron_options, num_neuron_options)) 

    # Define EarlyStopping callback for main models
    early_stopping_main = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

    pca_for_plotting = None # Initialize PCA object for plotting, if needed
    X_test_plot_data = X_test_scaled
    X_full_plot_data = X_full_scaled
    plot_feature_names = feature_cols

    if len(feature_cols) > 2:
        print("\nPreparing PCA for decision boundary visualization (fit on training data)...")
        pca_for_plotting = PCA(n_components=2, random_state=42)
        pca_for_plotting.fit(X_train_scaled) # Fit PCA on the scaled training data
        X_test_plot_data = pca_for_plotting.transform(X_test_scaled)
        X_full_plot_data = pca_for_plotting.transform(X_full_scaled)
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
                Dense(neurons_l1, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                Dense(neurons_l2, activation='relu'),
                Dense(1, activation='sigmoid')
            ])

            optimizer = Adam(learning_rate=0.01)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

            print(f"Training {full_model_description}...")
            history = model.fit(X_train_scaled, y_train_raw, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping_main])

            loss_test, accuracy_test = model.evaluate(X_test_scaled, y_test_raw, verbose=0)
            print(f"\n{full_model_description} - Test Loss: {loss_test:.4f}, Test Accuracy: {accuracy_test:.4f}")

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'{full_model_description}:\nModel Loss', fontsize=10)
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
            plt.ylim(0, 1) # Set y-axis limits for Loss plot

            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title(f'{full_model_description}:\nModel Accuracy', fontsize=10)
            plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
            plt.ylim(0, 1) # Set y-axis limits for Accuracy plot
            
            plt.tight_layout(rect=[0, 0, 1, 0.96]) 
            history_plot_filename = f'{model_name_prefix.lower().replace(" ", "_")}_training_history.png'
            plt.savefig(os.path.join(current_model_dir_path, history_plot_filename))
            plt.close()
            print(f"Training history plot saved to: {os.path.join(current_model_dir_path, history_plot_filename)}")

            # Decision Boundary Plotting Logic uses the main 'model'
            print(f"\nPlotting decision boundaries for {full_model_description}...")
            plot_decision_boundary(X_test_plot_data, y_test_raw, model, scaler, plot_feature_names, current_model_dir_path, "Test Set", full_model_description, pca_transformer=pca_for_plotting)
            plot_decision_boundary(X_full_plot_data, y_full_raw, model, scaler, plot_feature_names, current_model_dir_path, "Full Data", full_model_description, pca_transformer=pca_for_plotting)
                        
            # Model metrics are always based on the original model trained on original features
            plot_model_metrics(model, X_test_scaled, y_test_raw, X_train_scaled, y_train_raw, X_full_scaled, y_full_raw, feature_cols, current_model_dir_path, full_model_description)

            loss_full, accuracy_full = model.evaluate(X_full_scaled, y_full_raw, verbose=0)
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
parent_dir = os.path.join(current_dir, 'graficos', f'modelo_color_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
os.makedirs(parent_dir, exist_ok=True)

#directorio modelo simple
simple_dir = os.path.join(parent_dir, 'modelo_simple')
os.makedirs(simple_dir, exist_ok=True)

simple_nn_base_regression_dir = os.path.join(simple_dir, 'neural_network_and_lr_analysis_1_layer')
simple_nn_dir_2_layers = os.path.join(simple_dir, 'neural_network_2_layers')
os.makedirs(simple_nn_base_regression_dir, exist_ok=True)
os.makedirs(simple_nn_dir_2_layers, exist_ok=True)

#directorio modelo intermedio
intermediate_dir = os.path.join(parent_dir, 'modelo_intermedio')
os.makedirs(intermediate_dir, exist_ok=True)

intermediate_nn_base_regression_dir = os.path.join(intermediate_dir, 'neural_network_and_lr_analysis_1_layer')
intermediate_nn_dir_2_layers = os.path.join(intermediate_dir, 'neural_network_2_layers')
os.makedirs(intermediate_nn_base_regression_dir, exist_ok=True)
os.makedirs(intermediate_nn_dir_2_layers, exist_ok=True)


#directorio modelo completo
complete_dir = os.path.join(parent_dir, 'modelo_completo')
os.makedirs(complete_dir, exist_ok=True)

complete_nn_base_regression_dir = os.path.join(complete_dir, 'neural_network_and_lr_analysis_1_layer')
complete_nn_dir_2_layers = os.path.join(complete_dir, 'neural_network_2_layers')
os.makedirs(complete_nn_base_regression_dir, exist_ok=True)
os.makedirs(complete_nn_dir_2_layers, exist_ok=True)


#leer los datos
white_wine_data = pd.read_csv('TP 4 sommelier/winequality-white.csv', sep=';')
red_wine_data = pd.read_csv('TP 4 sommelier/winequality-red.csv', sep=';')

white_wine_data.name = 'white_wine_data'
red_wine_data.name = 'red_wine_data'

white_wine_data['type'] = 1 # 1 for white
red_wine_data['type'] = 0 # 0 for red

combined_data = pd.concat([white_wine_data, red_wine_data], axis=0)
print(combined_data.head())
combined_data.name = 'combined_data'





#MODELO SIMPLE 2 VARIABLES
feature_columns_simple = ['total sulfur dioxide', 'chlorides']
target_column = 'type'

scatterplot_vs_type(combined_data, feature_columns_simple[0], feature_columns_simple[1], simple_dir)

print("\n--- Simple model: Combined Neural Network and Logistic Regression Sensibility Analysis (1 Hidden Layer) ---")
neural_network_neurons_sensibility_analysis(combined_data, feature_columns_simple, target_column, simple_nn_base_regression_dir)

print("\n--- Simple model: Neural Network Sensibility Analysis (2 Hidden Layers) ---")
neural_network_two_hidden_layers_sensibility_analysis(combined_data, feature_columns_simple, target_column, simple_nn_dir_2_layers)






#MODELO INTERMEDIO 4 VARIABLES
feature_columns_intermediate = ['total sulfur dioxide', 'chlorides', 'volatile acidity', 'residual sugar']
target_column = 'type'

plot_pca(combined_data, feature_columns_intermediate, target_column, intermediate_dir)

print("\n--- Intermediate model: Combined Neural Network and Logistic Regression Sensibility Analysis (1 Hidden Layer) ---")
neural_network_neurons_sensibility_analysis(combined_data, feature_columns_intermediate, target_column, intermediate_nn_base_regression_dir)

print("\n--- Intermediate model: Neural Network Sensibility Analysis (2 Hidden Layers) ---")
neural_network_two_hidden_layers_sensibility_analysis(combined_data, feature_columns_intermediate, target_column, intermediate_nn_dir_2_layers)







#MODELO COMPLETO TODAS VARIABLES
feature_columns_complete = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
target_column = 'type'

plot_pca(combined_data, feature_columns_complete, target_column, complete_dir)

print("\n--- Complete model: Combined Neural Network and Logistic Regression Sensibility Analysis (1 Hidden Layer) ---")
neural_network_neurons_sensibility_analysis(combined_data, feature_columns_complete, target_column, complete_nn_base_regression_dir)

print("\n--- Complete model: Neural Network Sensibility Analysis (2 Hidden Layers) ---")
neural_network_two_hidden_layers_sensibility_analysis(combined_data, feature_columns_complete, target_column, complete_nn_dir_2_layers)





plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figures...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')