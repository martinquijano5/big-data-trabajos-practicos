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
from scipy.stats import mode
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from matplotlib.colors import ListedColormap

def float_to_int():
    print(df.info())
    
    for col in df.columns:
        df[col] = df[col].astype('int64')

def delete_duplicates():
    global df
    
    initial_shape = df.shape
    print(f"\nShape before removing duplicates: {initial_shape}")
    df = df.drop_duplicates()
    final_shape = df.shape
    print(f"Duplicates removed: {initial_shape[0] - final_shape[0]} rows deleted.")
    print(f"New shape after removing duplicates: {final_shape}")

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
        plt.close() # Close the figure to free memory

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

# Create directory for saving graphs
current_dir = os.path.dirname(os.path.abspath(__file__))
graficos_dir = os.path.join(current_dir, 'graficos', 'exploracion')
os.makedirs(graficos_dir, exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'proporciones'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'barchart_vs_diabetes'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'histogramas'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'boxplots'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'boxplots_vs_diabetes'), exist_ok=True)
os.makedirs(os.path.join(graficos_dir, 'correlation_matrix'), exist_ok=True)

# Load and prepare the dataset
df = pd.read_csv('punto 2/diabetes_binary_health_indicators_BRFSS2015.csv')

#df = df.head(100)

delete_duplicates()
float_to_int()

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



# Show all plots
plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')