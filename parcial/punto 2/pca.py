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

def pca_scatterplot(df, file_name):
    # Separate features (X) and target (y)
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']


    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

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



current_dir = os.path.dirname(os.path.abspath(__file__))
graficos_dir = os.path.join(current_dir, 'graficos', 'pca')
os.makedirs(graficos_dir, exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'pca_scatterplot'), exist_ok=True)

df = pd.read_csv('punto 2/diabetes_binary_health_indicators_BRFSS2015_transformed.csv')

pca_scatterplot(df, 'full_model')

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

    print(f"IV ranking table saved to: {save_path}")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    return iv_df


orden_iv = calculate_and_plot_iv(df) # Call the IV function (no warning suppression needed)


# Select specific features for df_top_5 based on your request
selected_features = [
    'HighBP', 'HighChol', 'DiffWalk',
    'BMI_Category_Normal', 'BMI_Category_Obese_III', 'BMI_Category_Obese_II',
    'BMI_Category_Obese_I', 'BMI_Category_Overweight',
    'GenHlth_4', 'GenHlth_2', 'GenHlth_5', 'GenHlth_3',
    'Diabetes_binary' # Include the target variable
]
df_top_5 = df[selected_features].copy()

pca_scatterplot(df_top_5, 'selected_features_model') 


# Show all plots
plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')