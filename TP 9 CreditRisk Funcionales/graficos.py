import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def float_to_int(df, cols):
    print(df.info())
    
    for col in cols:
        df[col] = df[col].astype('int64')
    print(df.info())

def prints(df):
    print("Head del dataset:")
    print(df.head())

    print("\nShape del dataset:")
    print(df.shape)

    print("\nDataset info:")
    print(df.info())

    print("\nDataset description:")
    print(df.describe())


    desc_stats = df.describe()


    fig, ax = plt.subplots(figsize=(36, 6))
    ax.axis('off')

    table = ax.table(
        cellText=desc_stats.round(2).values,
        rowLabels=desc_stats.index,
        colLabels=desc_stats.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f2f2f2']*len(desc_stats.columns),
        rowColours=['#f2f2f2']*len(desc_stats.index)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)

    plt.title(f'Descriptive Statistics', pad=20)

    plt.savefig(os.path.join(graficos_dir, f'descriptive_stats.png'), bbox_inches='tight')

def proportions(df, cols):
    for col in cols:
        plt.figure(figsize=(10, 8))
        
        # Calculate value counts and proportions
        value_counts = df[col].value_counts()
        value_proportions = df[col].value_counts(normalize=True)
        
        # Create labels with both count and proportion
        labels = [f'{val}\n(n={count}, {prop:.1%})' 
                 for val, count, prop in zip(value_counts.index, 
                                          value_counts.values, 
                                          value_proportions.values)]
        
        # Create pie chart
        plt.pie(value_counts.values, 
                labels=labels,
                autopct='',  # We don't need autopct since we have counts in labels
                startangle=90)
        
        plt.title(f'Proporción de valores para {col}')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Save the plot
        plt.savefig(os.path.join(graficos_dir, 'proporciones', f'proporciones_{col}.png'))

def barchart_vs_target(df, cols):
    
    for col in cols:
        plt.figure(figsize=(8, 6)) # Adjust size as needed
        
        # Create the countplot using seaborn
        sns.countplot(x=col, hue='Rechazo', data=df, palette='colorblind')
        
        # Set title and labels
        plt.title(f'Frecuencia de Rechazo para {col}')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=90)
        
        # Add legend
        plt.legend(title='Rechazo') # Updated legend title
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(graficos_dir, 'barchart_vs_target', f'barchart_vs_Rechazo_{col}.png'))

def histogramas(df, cols):
    for col in cols:
        plt.figure(figsize=(10, 6))
        plt.hist(df[col].dropna(), bins=30)
        plt.title(f'Histograma de {col}')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        plt.savefig(os.path.join(graficos_dir, 'histogramas', f'histograma_{col}.png'))

def hisogramas_vs_target(df, cols):
    for col in cols:
        plt.figure(figsize=(10, 6))
        
        # Filter data for each wine type
        rechazado = df[df['Rechazo'] == 1][col].dropna()
        no_rechazado = df[df['Rechazo'] == 0][col].dropna()
        
        # Plot histograms with histtype='step' to show outlines and avoid fill blending
        # Increase linewidth for better visibility of the steps
        plt.hist(rechazado, bins=30, label='Rechazado (1)', color='red', histtype='step', linewidth=1.5)
        plt.hist(no_rechazado, bins=30, label='No Rechazado (0)', color='blue', histtype='step', linewidth=1.5)
        
        plt.title(f'Histograma de {col} vs Rechazo')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        plt.legend(title='Rechazo') # Matplotlib's legend will pick up the labels
        
        # Create directory if it doesn't exist
        save_dir = os.path.join(graficos_dir, 'histogramas_vs_target')
        os.makedirs(save_dir, exist_ok=True)
        
        plt.savefig(os.path.join(save_dir, f'histograma_vs_Rechazo_{col}.png'))

def boxplots(df, cols):
    for col in cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=col, data=df)
        plt.title(f'Boxplot de {col}')
        plt.savefig(os.path.join(graficos_dir, 'boxplots', f'boxplot_{col}.png'))

def boxplots_vs_target(df, cols):
    for col in cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Rechazo', y=col, data=df)
        plt.title(f'Boxplot de {col} vs Rechazo')
        plt.savefig(os.path.join(graficos_dir, 'boxplots_vs_target', f'boxplot_vs_Rechazo_{col}.png'))

def correlation_matrix(df, use_mask=True):
    # Calculate the correlation matrix
    corr = df.corr()
    
    # Set up the matplotlib figure
    plt.figure(figsize=(18, 15)) # Adjust size as needed
    
    mask = None
    if use_mask:
        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        filename_suffix = '_masked'
        title_suffix = ' (Masked)'
    else:
        filename_suffix = '_full'
        title_suffix = ' (Full)'

    # Draw the heatmap
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f")
                
    plt.title(f'Matriz de Correlación{title_suffix}', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    
    # Save the plot
    plt.savefig(os.path.join(graficos_dir, 'correlation_matrix', f'correlation_matrix_heatmap{filename_suffix}.png'))

def correlation_vs_target(df, target_col, file_name):
    # Calculate the correlation with the target variable 'Diabetes_binary'
    corr_target = df.corr()[target_col].sort_values(ascending=False)
    
    # Remove the correlation of the target variable with itself
    corr_target = corr_target.drop(target_col)
    
    # Set up the matplotlib figure
    plt.figure(figsize=(15, 6)) # Adjust size as needed
    
    # Create the bar plot
    corr_target.plot(kind='bar', color='darkgoldenrod')
    
    # Add title and labels
    plt.title(f'Correlation with {target_col} for {file_name}')
    plt.xlabel('Features')
    plt.ylabel('Correlation Coefficient')
    
    # Add grid for better readability
    plt.grid(True)
    
    # Rotate x-axis labels
    plt.xticks(rotation=90)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(graficos_dir, 'correlation_matrix', f'{file_name}.png'))

def plot_woe_details_table(feature_name, woe_df, target_col_name, file_name_detail):
    """
    Plots a table of WOE details for a feature.
    Assumes woe_df has columns: 'Bin', 'Count_Good', 'Distr_Good', 'Count_Bad', 'Distr_Bad', 'WOE'
    """
    global graficos_dir
    save_dir = os.path.join(graficos_dir, 'woe_tables')
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, max(2, len(woe_df) * 0.5))) # Adjusted figsize
    ax.axis('tight')
    ax.axis('off')

    # Prepare cell data, formatting WOE, Distr_Good, Distr_Bad to 4 decimal places
    cell_data_for_table = []
    for i in range(len(woe_df)):
        row_values = woe_df.iloc[i].copy() # Use .copy() to avoid SettingWithCopyWarning
        row_values['WOE'] = f"{row_values['WOE']:.4f}"
        row_values['Distr_Good'] = f"{row_values['Distr_Good']:.4f}"
        row_values['Distr_Bad'] = f"{row_values['Distr_Bad']:.4f}"
        cell_data_for_table.append(row_values.tolist())

    table = ax.table(cellText=cell_data_for_table,
                     colLabels=woe_df.columns,
                     cellLoc='center',
                     loc='center',
                     colColours=['#E0E0E0']*len(woe_df.columns))
                    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.3)
    
    plt.title(f'WOE Details for {feature_name} (Target: {target_col_name})', pad=20, fontsize=14)
    plt.savefig(os.path.join(save_dir, f'woe_table_{file_name_detail}_{feature_name}.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)

def plot_woe_bar_chart(feature_name, woe_df, target_col_name, file_name_detail):
    """
    Plots a bar chart of WOE values for a feature.
    Assumes woe_df has columns: 'Bin', 'WOE'
    """
    global graficos_dir
    save_dir = os.path.join(graficos_dir, 'woe_bar_charts')
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ensure 'Bin' is treated as categorical for plotting order
    bin_labels = woe_df['Bin'].astype(str)
    woe_values = woe_df['WOE']

    ax.bar(bin_labels, woe_values, color='skyblue')
    
    plt.title(f'WOE per Bin for {feature_name} (Target: {target_col_name})', fontsize=14)
    plt.xlabel(f'Bins of {feature_name}')
    plt.ylabel('Weight of Evidence (WOE)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, f'woe_barchart_{file_name_detail}_{feature_name}.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)

def calculate_and_plot_iv(df, target_col, file_name_prefix):
    """
    Calculates Information Value (IV) for each feature in the DataFrame
    and plots a summary table. Also plots WOE details table and bar chart per feature.
    Assumes target_col is binary with 0 for 'Good' and 1 for 'Bad'.
    """
    global graficos_dir
    features_iv_summary = []
    all_features_woe_details = {} # To store WOE DataFrames for each feature

    feature_cols = [col for col in df.columns if col != target_col]
    epsilon = 1e-6  # Small epsilon to prevent log(0) or division by zero

    # Ensure target column is 0 and 1
    if not set(df[target_col].unique()).issubset({0, 1}):
        print(f"Target column {target_col} is not binary (0,1). Skipping IV/WOE calculation.")
        return pd.DataFrame(features_iv_summary), all_features_woe_details

    total_good_overall = df[df[target_col] == 0].shape[0]
    total_bad_overall = df[df[target_col] == 1].shape[0]

    if total_good_overall == 0 or total_bad_overall == 0:
        print("No 'Good' or 'Bad' observations in the target column. Skipping IV/WOE calculation.")
        return pd.DataFrame(features_iv_summary), all_features_woe_details
        
    for feature in feature_cols:
        df_temp = df[[feature, target_col]].copy()
        df_temp.dropna(subset=[feature], inplace=True)

        if df_temp.empty:
            features_iv_summary.append({'Feature': feature, 'IV': 0.0})
            all_features_woe_details[feature] = pd.DataFrame() # Empty DF
            continue

        feature_iv_total = 0.0
        current_feature_woe_bins_data = []

        # Determine if feature is continuous or categorical for binning strategy
        is_continuous_for_binning = pd.api.types.is_numeric_dtype(df_temp[feature]) and df_temp[feature].nunique() > 10
        
        grouped_data = None
        bin_edges_map = {} # For storing actual bin edges from qcut

        if is_continuous_for_binning:
            try:
                num_distinct_values = df_temp[feature].nunique()
                bins_q = min(10, num_distinct_values)
                
                if bins_q >= 2:
                    # labels=False gives integer labels from 0 to bins_q-1
                    df_temp['bin_int_label'], qcut_actual_edges = pd.qcut(df_temp[feature], q=bins_q, duplicates='drop', labels=False, retbins=True, precision=3)
                    grouped_data = df_temp.groupby('bin_int_label')
                    
                    for i in range(len(qcut_actual_edges) - 1):
                        bin_edges_map[i] = f"({qcut_actual_edges[i]:.2f}, {qcut_actual_edges[i+1]:.2f}]"
                else: # Not enough distinct values for qcut, treat as categorical
                    is_continuous_for_binning = False
            except ValueError: # Fallback if qcut fails
                is_continuous_for_binning = False
                
        if not is_continuous_for_binning: # Categorical or low-cardinality numeric
            grouped_data = df_temp.groupby(feature)

        if grouped_data:
            for group_key, group_df in grouped_data:
                
                bin_display_label = ""
                if is_continuous_for_binning:
                    bin_display_label = bin_edges_map.get(group_key, str(group_key))
                else: # Categorical
                    bin_display_label = str(group_key)

                count_good_in_bin = group_df[group_df[target_col] == 0].shape[0]
                count_bad_in_bin = group_df[group_df[target_col] == 1].shape[0]

                distr_good_in_bin = (count_good_in_bin / total_good_overall) if total_good_overall > 0 else 0
                distr_bad_in_bin = (count_bad_in_bin / total_bad_overall) if total_bad_overall > 0 else 0
                
                # WOE = ln( (DistrGood + eps) / (DistrBad + eps) )
                woe = np.log( (distr_good_in_bin + epsilon) / (distr_bad_in_bin + epsilon) )
                
                iv_for_group = (distr_good_in_bin - distr_bad_in_bin) * woe
                feature_iv_total += iv_for_group
                
                current_feature_woe_bins_data.append({
                    'Bin': bin_display_label,
                    'Count_Good': count_good_in_bin,
                    'Distr_Good': distr_good_in_bin,
                    'Count_Bad': count_bad_in_bin,
                    'Distr_Bad': distr_bad_in_bin,
                    'WOE': woe
                })
            
        woe_details_df_for_feature = pd.DataFrame(current_feature_woe_bins_data)
        all_features_woe_details[feature] = woe_details_df_for_feature
        features_iv_summary.append({'Feature': feature, 'IV': feature_iv_total})

        # Plot WOE details for the current feature
        if not woe_details_df_for_feature.empty:
            plot_woe_details_table(feature, woe_details_df_for_feature, target_col, file_name_prefix)
            plot_woe_bar_chart(feature, woe_details_df_for_feature, target_col, file_name_prefix)

    iv_summary_df = pd.DataFrame(features_iv_summary)
    iv_summary_df.sort_values(by='IV', ascending=False, inplace=True)
    iv_summary_df.reset_index(drop=True, inplace=True)
    
    # Plot IV summary table
    fig_iv, ax_iv = plt.subplots(figsize=(10, max(3, len(iv_summary_df) * 0.5))) # Adjusted figsize
    ax_iv.axis('tight')
    ax_iv.axis('off')
    
    cell_data_for_iv_table = []
    for i in range(len(iv_summary_df)):
        row_values = iv_summary_df.iloc[i].tolist()
        # Format IV to 4 decimal places
        try:
            iv_col_idx = iv_summary_df.columns.get_loc('IV')
            row_values[iv_col_idx] = f"{iv_summary_df.iloc[i]['IV']:.4f}"
        except KeyError: # Should not happen if IV column exists
            pass
        cell_data_for_iv_table.append(row_values)
        
    table_iv = ax_iv.table(cellText=cell_data_for_iv_table,
                           colLabels=iv_summary_df.columns,
                           cellLoc='center',
                           loc='center',
                           colColours=['#E0E0E0']*len(iv_summary_df.columns))
                    
    table_iv.auto_set_font_size(False)
    table_iv.set_fontsize(10)
    table_iv.scale(1.1, 1.3)
    
    plt.title(f'Information Value (IV) of Features (Target: {target_col}) - {file_name_prefix}', pad=20, fontsize=14)
    plt.savefig(os.path.join(graficos_dir, f'iv_summary_{file_name_prefix}.png'), bbox_inches='tight', dpi=150)
    plt.close(fig_iv)
    
    return iv_summary_df, all_features_woe_details

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

#directorio para guardar los graficos
current_dir = os.path.dirname(os.path.abspath(__file__))
graficos_dir = os.path.join(current_dir, 'graficos', 'exploracion')
os.makedirs(graficos_dir, exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'proporciones'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'barchart_vs_target'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'histogramas'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'histogramas_vs_target'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'boxplots'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'boxplots_vs_target'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'correlation_matrix'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'woe_tables'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'woe_bar_charts'), exist_ok=True)

#leer los datos
data = pd.read_excel('Base_Clientes Alemanes Transformed.xlsx')

data.name = 'data'

#float_to_int(data, ['columna x', 'columna y'])

#llamado a funciones

prints(data)

# Define categorical and numerical columns for plotting
target_column_name = 'Rechazo'
categorical_cols_for_plotting = []
numerical_cols_for_plotting = []

# Threshold for considering a numeric column as categorical based on unique values
# This is similar to the logic in calculate_and_plot_iv for binning strategy.
nunique_threshold_for_categorical = 10

for col in data.columns:
    if pd.api.types.is_object_dtype(data[col]):
        categorical_cols_for_plotting.append(col)
    elif pd.api.types.is_numeric_dtype(data[col]):
        if data[col].nunique() <= nunique_threshold_for_categorical:
            categorical_cols_for_plotting.append(col)
        else:
            numerical_cols_for_plotting.append(col)
    # Non-object and non-numeric types (like datetime if any) are currently ignored by this logic

print("\nCategorical columns identified for plotting:")
print(categorical_cols_for_plotting)
print("\nNumerical columns identified for plotting:")
print(numerical_cols_for_plotting)

# Plotting calls using the new categorization
if categorical_cols_for_plotting:
    proportions(data, categorical_cols_for_plotting)
    barchart_vs_target(data, categorical_cols_for_plotting) # Assumes target is 'Rechazo'

if numerical_cols_for_plotting:
    histogramas(data, numerical_cols_for_plotting)
    hisogramas_vs_target(data, numerical_cols_for_plotting) # Assumes target is 'Rechazo'
    boxplots(data, numerical_cols_for_plotting)
    boxplots_vs_target(data, numerical_cols_for_plotting) # Assumes target is 'Rechazo'

# Correlation and IV plots remain as they handle columns appropriately internally
correlation_matrix(data, use_mask=True)  # For the masked (triangular) version
correlation_matrix(data, use_mask=False) # For the full (square) version

correlation_vs_target(data, 'Rechazo', 'correlation_vs_Rechazo')

calculate_and_plot_iv(data, 'Rechazo', 'Rechazo')

plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')