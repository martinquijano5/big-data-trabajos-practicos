import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import squarify
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from scipy.stats import mode, ttest_ind, chi2_contingency
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from matplotlib.colors import ListedColormap
import itertools


def delete_duplicates():
    global df
    
    initial_shape = df.shape
    print(f"\nShape before removing duplicates: {initial_shape}")
    df = df.drop_duplicates()
    final_shape = df.shape
    print(f"Duplicates removed: {initial_shape[0] - final_shape[0]} rows deleted.")
    print(f"New shape after removing duplicates: {final_shape}")

def float_to_int():
    print(df.info())
    
    for col in df.columns:
        df[col] = df[col].astype('int64')

def sacar_outliers():
    global df
    print("\n--- Removing outliers using IQR method ---")
    initial_shape = df.shape
    print(f"Shape before removing outliers: {initial_shape}")

    # Identify columns that are not binary (likely continuous or ordinal)
    # It's generally better to pass this as an argument or calculate it reliably
    # For now, recalculating based on current df state
    other_cols = df.columns[df.nunique() > 2]
    print(f"Columns being checked for outliers: {other_cols.tolist()}")

    total_outliers_removed = 0

    for col in other_cols:
        shape_before_col = df.shape[0]
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        low_lim = q1 - 1.5 * iqr
        up_lim = q3 + 1.5 * iqr

        # Filter the DataFrame
        df = df[(df[col] >= low_lim) & (df[col] <= up_lim)]

        shape_after_col = df.shape[0]
        outliers_removed_col = shape_before_col - shape_after_col
        if outliers_removed_col > 0:
            print(f"Outliers removed for column '{col}': {outliers_removed_col}")
            total_outliers_removed += outliers_removed_col

    final_shape = df.shape
    print(f"\nTotal outliers removed across all checked columns: {initial_shape[0] - final_shape[0]}")
    print(f"New shape after removing outliers: {final_shape}")

def prints():
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

    plt.title('Descriptive Statistics', pad=20)

    plt.savefig(os.path.join(graficos_dir, 'descriptive_stats.png'), bbox_inches='tight')

def analyze_value_distribution():
    print("\nValue distribution per column:")
    # Store the original display option
    original_max_rows = pd.get_option('display.max_rows')
    # Set option to display all rows
    pd.set_option('display.max_rows', None)

    for col in df.columns:
        print(f"\n--- Column: {col} ---")
        # Calculate value counts and proportions
        value_counts = df[col].value_counts()
        value_proportions = df[col].value_counts(normalize=True)

        # Combine into a DataFrame for display
        distribution_df = pd.DataFrame({
            'Count': value_counts,
            'Proportion': value_proportions.round(4) # Round for readability
        })
        print(distribution_df)

    # Reset the display option to its original value
    pd.set_option('display.max_rows', original_max_rows)

def proportions(cols):
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

def barchart_vs_diabetes(cols):
    # Map numeric Diabetes_binary to string labels for clarity in plots
    # Create a temporary column to avoid modifying the original df permanently within loop iterations
    df_temp = df.copy()
    df_temp['Diabetes_binary_str'] = df_temp['Diabetes_binary'].map({0: 'NON-Diabetic', 1: 'Diabetic'})
    
    for col in cols:
        # Skip the target variable itself if it's in the list
        if col == 'Diabetes_binary':
            continue
            
        plt.figure(figsize=(8, 6)) # Adjust size as needed
        
        # Create the countplot using seaborn
        sns.countplot(x=col, hue='Diabetes_binary_str', data=df_temp, palette='colorblind')
        
        # Set title and labels
        plt.title(f'Diabetes Disease Frequency for {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=90)
        
        # Add legend
        plt.legend(title='Diabetes Status') # Updated legend title
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(graficos_dir, 'barchart_vs_diabetes', f'barchart_vs_diabetes_{col}.png'))

def histogramas(cols):
    for col in cols:
        plt.figure(figsize=(10, 6))
        plt.hist(df[col].dropna(), bins=30)
        plt.title(f'Histograma de {col}')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        plt.savefig(os.path.join(graficos_dir, 'histogramas', f'histograma_{col}.png'))

def boxplots(cols):
    for col in cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=col, data=df)
        plt.title(f'Boxplot de {col}')
        plt.savefig(os.path.join(graficos_dir, 'boxplots', f'boxplot_{col}.png'))

def boxplots_vs_diabetes(cols):
    for col in cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Diabetes_binary', y=col, data=df)
        plt.title(f'Boxplot de {col} vs Diabetes')
        plt.savefig(os.path.join(graficos_dir, 'boxplots_vs_diabetes', f'boxplot_vs_diabetes_{col}.png'))

def correlation_matrix():
    # Calculate the correlation matrix
    corr = df.corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Set up the matplotlib figure
    plt.figure(figsize=(18, 15)) # Adjust size as needed
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f")
                
    plt.title('Matriz de Correlación', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    
    # Save the plot
    plt.savefig(os.path.join(graficos_dir, 'correlation_matrix', 'correlation_matrix_heatmap.png'))

def correlation_vs_diabetes():
    # Calculate the correlation with the target variable 'Diabetes_binary'
    corr_target = df.corr()['Diabetes_binary'].sort_values(ascending=False)
    
    # Remove the correlation of the target variable with itself
    corr_target = corr_target.drop('Diabetes_binary')
    
    # Set up the matplotlib figure
    plt.figure(figsize=(15, 6)) # Adjust size as needed
    
    # Create the bar plot
    corr_target.plot(kind='bar', color='darkgoldenrod')
    
    # Add title and labels
    plt.title('Correlation with Diabetes_binary')
    plt.xlabel('Features')
    plt.ylabel('Correlation Coefficient')
    
    # Add grid for better readability
    plt.grid(True)
    
    # Rotate x-axis labels
    plt.xticks(rotation=90)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(graficos_dir, 'correlation_matrix', 'correlation_vs_diabetes.png'))

def spearman_correlation():
    # Calculate the Spearman correlation matrix
    corr_spearman = df.corr(method='spearman')

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_spearman, dtype=bool))

    # Set up the matplotlib figure
    plt.figure(figsize=(18, 15)) # Adjust size as needed

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_spearman, mask=mask, cmap='coolwarm', vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f")

    plt.title('Matriz de Correlación (Spearman)', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout() # Adjust layout to prevent labels overlapping

    # Save the plot
    plt.savefig(os.path.join(graficos_dir, 'correlation_matrix', 'correlation_matrix_spearman_heatmap.png'))

def spearman_correlation_vs_diabetes():
    # Calculate the Spearman correlation with the target variable 'Diabetes_binary'
    corr_target_spearman = df.corr(method='spearman')['Diabetes_binary'].sort_values(ascending=False)

    # Remove the correlation of the target variable with itself
    corr_target_spearman = corr_target_spearman.drop('Diabetes_binary')

    # Set up the matplotlib figure
    plt.figure(figsize=(15, 6)) # Adjust size as needed

    # Create the bar plot
    corr_target_spearman.plot(kind='bar', color='mediumseagreen') # Changed color slightly for distinction

    # Add title and labels
    plt.title('Spearman Correlation with Diabetes_binary')
    plt.xlabel('Features')
    plt.ylabel('Spearman Correlation Coefficient')

    # Add grid for better readability
    plt.grid(True)

    # Rotate x-axis labels
    plt.xticks(rotation=90)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(graficos_dir, 'correlation_matrix', 'spearman_correlation_vs_diabetes.png'))

def perform_t_tests_vs_diabetes(df, continuous_cols, target_col='Diabetes_binary', alpha=0.05):
    """
    Performs independent samples t-tests for continuous columns against a binary target variable
    and saves the results as a table image.

    Args:
        df (pd.DataFrame): The input DataFrame.
        continuous_cols (list): List of continuous/ordinal column names to test.
        target_col (str): The name of the binary target column.
        alpha (float): The significance level for interpreting p-values.
    """
    print(f"\n--- Performing Independent Samples T-tests vs {target_col} (alpha={alpha}) ---")

    group0 = df[df[target_col] == 0]
    group1 = df[df[target_col] == 1]

    results = []

    for col in continuous_cols:

        # Perform t-test, ignoring NaNs
        stat, p_value = ttest_ind(group0[col], group1[col], equal_var=False) # Welch's t-test

        significant = "Yes" if p_value < alpha else "No"
        results.append({
            'Variable': col,
            'T-Statistic': f"{stat:.4f}",
            'P-Value': f"{p_value:.4f}",
            f'Significant (p < {alpha})': significant
        })


    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Create table plot
    fig, ax = plt.subplots(figsize=(10, len(results_df) * 0.5 + 1)) # Adjust size based on number of rows
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=results_df.values,
                       colLabels=results_df.columns,
                       cellLoc='center',
                       loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2) # Adjust scale as needed

    plt.title(f'T-Test Results: Features vs {target_col}', pad=20, fontsize=14)

    # Define save path
    save_dir = os.path.join(graficos_dir, 't_tests_vs_diabetes')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f't_test_results_{target_col}.png')

    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"T-test results table saved to: {save_path}")
    
def perform_chi2_tests_vs_diabetes(df, binary_cols, target_col='Diabetes_binary', alpha=0.05):
    """
    Performs Chi-squared tests of independence for binary columns against a binary target variable
    and saves the results as a table image.

    Args:
        df (pd.DataFrame): The input DataFrame.
        binary_cols (list): List of binary column names to test.
        target_col (str): The name of the binary target column.
        alpha (float): The significance level for interpreting p-values.
    """
    print(f"\n--- Performing Chi-squared Tests vs {target_col} (alpha={alpha}) ---")

    results = []

    for col in binary_cols:
        if col == target_col: # Skip the target column itself
            continue

        # Create contingency table
        contingency_table = pd.crosstab(df[col], df[target_col])

        # Perform Chi-squared test

        # The test returns chi2 stat, p-value, degrees of freedom, expected frequencies
        chi2_stat, p_value, dof, expected_freq = chi2_contingency(contingency_table)

        significant = "Yes" if p_value < alpha else "No"
        results.append({
            'Variable': col,
            'Chi2-Statistic': f"{chi2_stat:.4f}",
            'P-Value': f"{p_value:.4e}", # Use scientific notation for potentially very small p-values
            'Degrees of Freedom': dof,
            f'Significant (p < {alpha})': significant
        })
        # Optional: Print summary
        # print(f"\nVariable: {col} | Chi2: {chi2_stat:.4f} | P-value: {p_value:.4e} | Significant: {significant}")


    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Create table plot
    fig, ax = plt.subplots(figsize=(10, len(results_df) * 0.5 + 1)) # Adjust size
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=results_df.values,
                       colLabels=results_df.columns,
                       cellLoc='center',
                       loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2) # Adjust scale

    plt.title(f'Chi-squared Test Results: Binary Features vs {target_col}', pad=20, fontsize=14)

    # Define save path
    save_dir = os.path.join(graficos_dir, 'chi2_tests_vs_diabetes')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'chi2_test_results_{target_col}.png')

    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Chi-squared test results table saved to: {save_path}")

def calculate_and_plot_iv(df, target_col='Diabetes_binary', bins=10):
    """
    Calculates Weight of Evidence (WoE) and Information Value (IV) for features
    and generates a table ranking features by IV. Handles both categorical and
    continuous features (by binning). Replaces 0 distribution values with 1e-6
    to avoid log(0) errors. Colors rows based on IV strength.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the binary target column (0s and 1s).
        bins (int): Number of bins to use for continuous variables.
    """
    print(f"\n--- Calculating WoE and IV for Features vs {target_col} ---")

    # Separate features and target
    y = df[target_col]
    X = df.drop(columns=[target_col])

    iv_values = {}
    epsilon = 1e-6 # Small value to replace zero distributions

    # Calculate total number of events (1s) and non-events (0s)
    total_event = y.sum()
    total_non_event = len(y) - total_event


    for col in X.columns:
        feature_iv = 0
        feature_df = pd.DataFrame({'feature': X[col], 'target': y})

        # Decide binning strategy: Use unique values if fewer than 'bins', otherwise qcut
        if feature_df['feature'].nunique() <= bins:
             # Treat as categorical or discrete with few values
             feature_df['feature_binned'] = feature_df['feature']
        else:
            # Bin continuous or high-cardinality features
            # Use labels=False to get bin identifiers, handles potential non-unique edges
            feature_df['feature_binned'] = pd.qcut(feature_df['feature'], q=bins, labels=False, duplicates='drop')


        # Group by the binned/categorical feature
        grouped = feature_df.groupby('feature_binned')['target']
        stats = grouped.agg(['count', 'sum']) # count = total in bin, sum = number of events (1s)
        stats.rename(columns={'count': 'total_count', 'sum': 'event_count'}, inplace=True)

        # Calculate non-event count
        stats['non_event_count'] = stats['total_count'] - stats['event_count']

        # Calculate distributions
        stats['event_dist'] = stats['event_count'] / total_event
        stats['non_event_dist'] = stats['non_event_count'] / total_non_event

        # Replace 0s with epsilon before calculating WoE
        stats['event_dist'] = stats['event_dist'].replace(0, epsilon)
        stats['non_event_dist'] = stats['non_event_dist'].replace(0, epsilon)

        # Calculate WoE
        # WoE = ln( Distr_Non_Event / Distr_Event )
        stats['woe'] = np.log(stats['non_event_dist'] / stats['event_dist'])

        # Calculate IV for the bin
        # IV = (Distr_Non_Event - Distr_Event) * WoE
        stats['iv'] = (stats['non_event_dist'] - stats['event_dist']) * stats['woe']

        # Sum IV across bins for the feature
        feature_iv = stats['iv'].sum()
        iv_values[col] = feature_iv


    # Create DataFrame for results
    iv_df = pd.DataFrame(list(iv_values.items()), columns=['Feature', 'InformationValue'])
    iv_df = iv_df.sort_values(by='InformationValue', ascending=False).reset_index(drop=True)

    # --- Define Colors based on IV ---


    # --- Define Colors based on IV (Gray -> Yellow -> Green) ---
    row_colors = [] # This will store the single color for each row
    """
    https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html#what_is_information_value
    If the IV statistic is:
    Less than 0.02, then the predictor is not useful for modeling (separating the Goods from the Bads)
    0.02 to 0.1, then the predictor has only a weak relationship to the Goods/Bads odds ratio
    0.1 to 0.3, then the predictor has a medium strength relationship to the Goods/Bads odds ratio
    0.3 to 0.5, then the predictor has a strong relationship to the Goods/Bads odds ratio.
    > 0.5, suspicious relationship (Check once)
    """
    for iv in iv_df['InformationValue']:
        if iv < 0.02:
            row_colors.append('#f0f0f0') # Light Gray - Not useful
        elif 0.02 <= iv < 0.1:
            row_colors.append('#ffffcc') # Pale Yellow - Weak
        elif 0.1 <= iv < 0.3:
            row_colors.append('#e0ffe0') # Pale Green - Medium
        elif 0.3 <= iv < 0.5:
            row_colors.append('#a0ffa0') # Light Green - Strong
        else: # iv >= 0.5
            row_colors.append('#60ff60') # Green - Suspicious/Very Strong

    # Format IV values for the table
    table_data = iv_df.copy()
    table_data['InformationValue'] = table_data['InformationValue'].map('{:.6f}'.format)

    # --- Create cellColours array ---
    # It needs to be the same shape as the data
    cell_colors = []
    num_cols = table_data.shape[1]
    for r_color in row_colors:
        cell_colors.append([r_color] * num_cols) # Repeat row color for each cell in the row


    # --- Create Table Plot ---
    fig, ax = plt.subplots(figsize=(8, len(iv_df) * 0.4 + 1)) # Adjust size
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=table_data.values,
                       colLabels=iv_df.columns,
                       cellColours=cell_colors, # Use cellColours instead of rowColours
                       cellLoc='center',
                       loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.2) # Adjust scale

    plt.title('Feature Importance Ranking by Information Value (IV)', pad=20, fontsize=14)

    # Define save path
    save_dir = os.path.join(graficos_dir, 'woe_iv')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'iv_ranking_table.png')

    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"IV ranking table saved to: {save_path}")

    return iv_df

def pca_scatterplot(df, file_name):
    # Separate features (X) and target (y)
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Create a DataFrame with PCA results
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['Diabetes_binary'] = y.values # Add target variable for coloring

    # --- Explained Variance and Components Table ---
    explained_variance_ratio = pca.explained_variance_ratio_
    components = pca.components_

    # Create a DataFrame for the components table (loadings)
    components_df = pd.DataFrame(components.T, columns=['PC1', 'PC2'], index=X.columns)

    # Create a separate DataFrame for the explained variance row
    variance_row = pd.DataFrame([explained_variance_ratio], columns=['PC1', 'PC2'], index=['Explained Variance Ratio'])

    # Concatenate the loadings and the variance row
    components_table_df = pd.concat([components_df, variance_row])


    # Create the table plot
    fig_table, ax_table = plt.subplots(figsize=(12, 8)) # Adjust size as needed
    ax_table.axis('tight')
    ax_table.axis('off')
    # Use the new components_table_df which includes the variance row correctly
    table = ax_table.table(cellText=components_table_df.round(3).values,
                           colLabels=['PC1', 'PC2'], # Explicitly set column labels
                           rowLabels=components_table_df.index,
                           cellLoc = 'center',
                           loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title(f'PCA Components Loadings and Explained Variance Ratio for {file_name}', pad=20)
    plt.savefig(os.path.join(graficos_dir, 'pca_scatterplot', f'{file_name}_pca_components_table.png'), bbox_inches='tight')

    # --- Scatter Plot ---
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(x='PC1', y='PC2', hue='Diabetes_binary', data=pca_df,
                              palette={0: 'blue', 1: 'red'}, alpha=0.7)
    plt.title(f'PCA Scatter Plot (PC1 vs PC2) Colored by Diabetes Status for {file_name}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    handles, labels = scatter.get_legend_handles_labels()
    scatter.legend(handles=handles, labels=['Non-Diabetic', 'Diabetic'], title='Diabetes Status')
    plt.grid(True)
    plt.savefig(os.path.join(graficos_dir, 'pca_scatterplot', f'{file_name}_pca_scatterplot.png'))

def pca_top_x(orden_iv, cant_features):
    # Get the top x features based on IV
    top_x_features = orden_iv['Feature'].head(cant_features).tolist()

    # Create a new DataFrame including the top 6 features AND the target variable
    features_to_include = top_x_features + ['Diabetes_binary']
    top_x_df = df[features_to_include].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Now call pca_scatterplot with the correctly formed DataFrame
    pca_scatterplot(top_x_df, f'top_{cant_features}_model')

def simple_scatterplot(var1, var2):
    """
    Creates a scatter plot of two variables colored by Diabetes_binary status.

    Args:
        var1 (str): The name of the column for the x-axis.
        var2 (str): The name of the column for the y-axis.
    """
    # Ensure the directory for saving exists
    save_dir = os.path.join(graficos_dir, 'simple_scatterplots')
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(x=var1, y=var2, hue='Diabetes_binary', data=df,
                              palette={0: 'blue', 1: 'red'}, alpha=0.7) # Use same palette as PCA

    plt.title(f'Scatter Plot: {var1} vs {var2} by Diabetes Status')
    plt.xlabel(var1)
    plt.ylabel(var2)

    # Customize legend
    handles, labels = scatter.get_legend_handles_labels()
    scatter.legend(handles=handles, labels=['Non-Diabetic', 'Diabetic'], title='Diabetes Status')

    plt.grid(True)

    # Define save path
    save_path = os.path.join(save_dir, f'scatter_{var1}_vs_{var2}.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Simple scatter plot saved to: {save_path}")

def simple_scatterplot_combinations():
    """
    Generates scatter plots for all unique combinations of two variables
    in the DataFrame, colored by Diabetes_binary status.
    """
    global df # Access the global DataFrame
    cols = df.columns.tolist()

    # Remove the target variable from the list if we don't want plots like 'BMI vs Diabetes_binary'
    # If you *do* want those, keep the original 'cols' list.
    cols_to_plot = [col for col in cols if col != 'Diabetes_binary']

    # Generate all unique combinations of 2 columns
    for var1, var2 in itertools.combinations(cols_to_plot, 2):
        simple_scatterplot(var1, var2)

# Create directory for saving graphs
current_dir = os.path.dirname(os.path.abspath(__file__))
graficos_dir = os.path.join(current_dir, 'graficos', 'exploracion_sin_outliers')
os.makedirs(graficos_dir, exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'proporciones'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'barchart_vs_diabetes'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'histogramas'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'boxplots'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'boxplots_vs_diabetes'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'correlation_matrix'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 't_tests_vs_diabetes'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'chi2_tests_vs_diabetes'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'woe_iv'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'pca_scatterplot'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'simple_scatterplots'), exist_ok=True)


# Load and prepare the dataset
df = pd.read_csv('punto 2/diabetes_binary_health_indicators_BRFSS2015.csv')

#df = df.head(100)

delete_duplicates()
float_to_int()

sacar_outliers()

prints()
#analyze_value_distribution()
binary_cols = df.columns[df.nunique() == 2]
print(binary_cols)

proportions(binary_cols)
barchart_vs_diabetes(binary_cols)

other_cols = df.columns[df.nunique() > 2]
print(other_cols)


histogramas(other_cols)
boxplots(other_cols)
boxplots_vs_diabetes(other_cols)


correlation_matrix()
correlation_vs_diabetes()


spearman_correlation()
spearman_correlation_vs_diabetes()


perform_t_tests_vs_diabetes(df, other_cols)
perform_chi2_tests_vs_diabetes(df, binary_cols)


orden_iv = calculate_and_plot_iv(df) # Call the IV function (no warning suppression needed)


pca_scatterplot(df, 'full_model')

pca_top_x(orden_iv, 6)

pca_top_x(orden_iv, 4)

pca_top_x(orden_iv, 3)

pca_top_x(orden_iv, 8)

pca_top_x(orden_iv, 7)

pca_top_x(orden_iv, 5)

# Generate scatter plots for all combinations
#simple_scatterplot_combinations()


# Show all plots
plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')