import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def float_to_int():
    print(combined_data.info())
    
    for col in ['free sulfur dioxide', 'total sulfur dioxide']:
        combined_data[col] = combined_data[col].astype('int64')
    print(combined_data.info())

def prints(df, nombre):
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

    plt.title(f'Descriptive Statistics - {nombre}', pad=20)

    plt.savefig(os.path.join(graficos_dir, f'descriptive_stats_{nombre}.png'), bbox_inches='tight')

def proportions(cols):
    for col in cols:
        plt.figure(figsize=(10, 8))
        
        # Calculate value counts and proportions
        value_counts = combined_data[col].value_counts()
        value_proportions = combined_data[col].value_counts(normalize=True)
        
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

def barchart_vs_type(cols):
    
    for col in cols:
        # Skip the target variable itself if it's in the list
        if col == 'type':
            continue
            
        plt.figure(figsize=(8, 6)) # Adjust size as needed
        
        # Create the countplot using seaborn
        sns.countplot(x=col, hue='type', data=combined_data, palette='colorblind')
        
        # Set title and labels
        plt.title(f'Wine Type Frequency for {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=90)
        
        # Add legend
        plt.legend(title='Wine Type') # Updated legend title
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(graficos_dir, 'barchart_vs_type', f'barchart_vs_type_{col}.png'))

def histogramas(cols):
    for col in cols:
        plt.figure(figsize=(10, 6))
        plt.hist(combined_data[col].dropna(), bins=30)
        plt.title(f'Histograma de {col}')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        plt.savefig(os.path.join(graficos_dir, 'histogramas', f'histograma_{col}.png'))

def hisogramas_vs_type(cols):
    for col in cols:
        # Skip the target variable itself if it's in the list
        if col == 'type':
            continue

        plt.figure(figsize=(10, 6))
        
        # Filter data for each wine type
        white_wine = combined_data[combined_data['type'] == 1][col].dropna()
        red_wine = combined_data[combined_data['type'] == 0][col].dropna()
        
        # Plot histograms with histtype='step' to show outlines and avoid fill blending
        # Increase linewidth for better visibility of the steps
        plt.hist(white_wine, bins=30, label='White Wine', color='blue', histtype='step', linewidth=1.5)
        plt.hist(red_wine, bins=30, label='Red Wine', color='red', histtype='step', linewidth=1.5)
        
        plt.title(f'Histograma de {col} vs Wine Type')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        plt.legend(title='Wine Type') # Matplotlib's legend will pick up the labels
        
        # Create directory if it doesn't exist
        save_dir = os.path.join(graficos_dir, 'histogramas_vs_type')
        os.makedirs(save_dir, exist_ok=True)
        
        plt.savefig(os.path.join(save_dir, f'histograma_vs_type_{col}.png'))

def boxplots(cols):
    for col in cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=col, data=combined_data)
        plt.title(f'Boxplot de {col}')
        plt.savefig(os.path.join(graficos_dir, 'boxplots', f'boxplot_{col}.png'))

def boxplots_vs_type(cols):
    for col in cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='type', y=col, data=combined_data)
        plt.title(f'Boxplot de {col} vs Wine Type')
        plt.savefig(os.path.join(graficos_dir, 'boxplots_vs_type', f'boxplot_vs_type_{col}.png'))

def correlation_matrix(use_mask=True):
    # Calculate the correlation matrix
    corr = combined_data.corr()
    
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

def correlation_vs_type(df, target_col, file_name):
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

def calculate_and_plot_iv(df, target_col, file_name):
    """
    Calculates Information Value (IV) for each feature in the DataFrame
    and plots a summary table. Handles both binary and multi-class targets.
    """
    global graficos_dir
    features_iv = []
    feature_cols = [col for col in df.columns if col != target_col]
    
    # Check if target is binary or multi-class
    unique_values = df[target_col].unique()
    is_binary = len(unique_values) == 2
    
    # For multi-class, we'll use one-vs-rest approach
    if is_binary:
        target_classes = [0, 1]  # Binary case: assume 0 and 1
    else:
        target_classes = unique_values  # All unique values
    
    epsilon = 0.00001  # To avoid log(0) or division by zero

    # Calculate IV for each feature
    for feature in feature_cols:
        df_temp = df[[feature, target_col]].copy()
        df_temp.dropna(subset=[feature], inplace=True)
        
        if df_temp.empty:
            features_iv.append({'Feature': feature, 'IV': 0.0})
            continue
            
        # Calculate overall IV for this feature across all target classes
        overall_iv = 0.0
        
        # For multiclass, calculate IV for each class and take the maximum
        class_ivs = []
        
        for target_class in target_classes:
            # Create a binary version of the target for this class
            df_temp['binary_target'] = (df_temp[target_col] == target_class).astype(int)
            
            total_events = df_temp[df_temp['binary_target'] == 0].shape[0]
            total_non_events = df_temp[df_temp['binary_target'] == 1].shape[0]
            
            if total_events == 0 or total_non_events == 0:
                continue  # Skip this class if there are no events or non-events
                
            feature_iv_total = 0.0
            
            # Handle constant features
            if df_temp[feature].nunique() == 1:
                feature_iv_total = 0.0
            else:
                # Determine if feature is continuous or categorical for binning strategy
                is_continuous_for_binning = pd.api.types.is_numeric_dtype(df_temp[feature]) and df_temp[feature].nunique() > 10
                
                grouped_data = None
                
                if is_continuous_for_binning:
                    try:
                        num_distinct_values = df_temp[feature].nunique()
                        # Number of bins: min of 10, or number of distinct values if less than 10
                        bins_q = min(10, num_distinct_values)
                        
                        if bins_q >= 2:
                            df_temp['bin'] = pd.qcut(df_temp[feature], q=bins_q, duplicates='drop', labels=False)
                            grouped_data = df_temp.groupby('bin')
                        else:
                            is_continuous_for_binning = False
                    except ValueError:
                        is_continuous_for_binning = False
                        
                if not is_continuous_for_binning:
                    grouped_data = df_temp.groupby(feature)
                    
                if grouped_data:
                    for _, group in grouped_data:
                        events_in_group = group[group['binary_target'] == 0].shape[0]
                        non_events_in_group = group[group['binary_target'] == 1].shape[0]
                        
                        if total_events > 0:
                            dist_events = events_in_group / total_events
                        else:
                            dist_events = 0
                            
                        if total_non_events > 0:
                            dist_non_events = non_events_in_group / total_non_events
                        else:
                            dist_non_events = 0
                            
                        # Adjust with epsilon for WOE calculation
                        dist_events_adj = max(dist_events, epsilon)
                        dist_non_events_adj = max(dist_non_events, epsilon)
                        
                        woe = np.log(dist_non_events_adj / dist_events_adj)
                        
                        # IV for the group
                        iv_for_group = (dist_non_events - dist_events) * woe
                        feature_iv_total += iv_for_group
                        
            class_ivs.append(feature_iv_total)
            
        # For multi-class, use the maximum IV across all classes
        # For binary, this will be the same as the original IV
        if class_ivs:
            overall_iv = max(class_ivs)
            
        features_iv.append({'Feature': feature, 'IV': overall_iv})
        
    # Rest of your function remains the same
    iv_df = pd.DataFrame(features_iv)
    iv_df.sort_values(by='IV', ascending=False, inplace=True)
    iv_df.reset_index(drop=True, inplace=True)
    
    # Plotting code remains the same
    fig, ax = plt.subplots(figsize=(12, max(4, len(iv_df) * 0.6)))
    ax.axis('tight')
    ax.axis('off')
    
    cell_data_for_table = []
    for i in range(len(iv_df)):
        row_values = iv_df.iloc[i].tolist()
        try:
            iv_col_idx = iv_df.columns.get_loc('IV')
            row_values[iv_col_idx] = f"{iv_df.iloc[i]['IV']:.4f}"
        except KeyError:
            pass
        cell_data_for_table.append(row_values)
        
    table = ax.table(cellText=cell_data_for_table,
                    colLabels=iv_df.columns,
                    cellLoc='center',
                    loc='center',
                    # cellColours=cell_colors_for_table, # Removed cell coloring
                    colColours=['#E0E0E0']*len(iv_df.columns))
                    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.3)
    
    plt.title(f'Information Value (IV) of Features for {file_name}', pad=20, fontsize=16)
    plt.savefig(os.path.join(graficos_dir, f'{file_name}.png'), bbox_inches='tight', dpi=150)
    
    return iv_df

#directorio para guardar los graficos
current_dir = os.path.dirname(os.path.abspath(__file__))
graficos_dir = os.path.join(current_dir, 'graficos', 'exploracion')
os.makedirs(graficos_dir, exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'proporciones'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'barchart_vs_type'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'histogramas'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'histogramas_vs_type'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'boxplots'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'boxplots_vs_type'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'correlation_matrix'), exist_ok=True)

#leer los datos
white_wine_data = pd.read_csv('TP 4 sommelier/winequality-white.csv', sep=';')
red_wine_data = pd.read_csv('TP 4 sommelier/winequality-red.csv', sep=';')

white_wine_data.name = 'white_wine_data'
red_wine_data.name = 'red_wine_data'

#agregar la columna tipo de vino
white_wine_data['type'] = 1 # 1 for white
red_wine_data['type'] = 0 # 0 for red

#combinar los dataframes
combined_data = pd.concat([white_wine_data, red_wine_data], axis=0)
print(combined_data.head())
combined_data.name = 'combined_data'


float_to_int()

#llamado a funciones

prints(white_wine_data, 'white_wine_data')
prints(red_wine_data, 'red_wine_data')

prints(combined_data, 'combined_data')

binary_cols = combined_data.columns[combined_data.nunique() == 2]
print(binary_cols)

proportions(binary_cols)
barchart_vs_type(binary_cols)

other_cols = combined_data.columns[combined_data.nunique() > 2]
print(other_cols)

histogramas(other_cols)
hisogramas_vs_type(other_cols)

boxplots(other_cols)
boxplots_vs_type(other_cols)

correlation_matrix(use_mask=True)  # For the masked (triangular) version
correlation_matrix(use_mask=False) # For the full (square) version

correlation_vs_type(combined_data, 'type', 'correlation_vs_type')
correlation_vs_type(white_wine_data.drop(columns=['type']), 'quality', 'correlation_vs_quality_white')
correlation_vs_type(red_wine_data.drop(columns=['type']), 'quality', 'correlation_vs_quality_red')

calculate_and_plot_iv(combined_data.drop(columns=['quality']), 'type', 'information_value_combined_type')
calculate_and_plot_iv(combined_data, 'quality', 'information_value_combined_quality')

calculate_and_plot_iv(white_wine_data.drop(columns=['type']), 'quality', 'information_value_white_quality')
calculate_and_plot_iv(red_wine_data.drop(columns=['type']), 'quality', 'information_value_red_quality')

plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')