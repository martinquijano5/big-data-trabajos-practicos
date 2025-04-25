import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
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

def prints(df):
    print("Head of the dataset:")
    print(df.head())

    print("\nDataset information:")
    print(df.info())

    print("\nDataset description:")
    print(df.describe())

    # Get descriptive statistics
    desc_stats = df.describe()
    
    # Create figure and axis for the table
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Hide axes
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=desc_stats.round(2).values,
        rowLabels=desc_stats.index,
        colLabels=desc_stats.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f2f2f2']*len(desc_stats.columns),
        rowColours=['#f2f2f2']*len(desc_stats.index)
    )
    
    # Adjust table style
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Add title
    plt.title('Descriptive Statistics', pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(graficos_dir, 'descriptive_stats.png'))

def plot_z_distribution(df):
    z_counts = df['z'].value_counts()
    plt.figure(figsize=(10, 6))
    plt.pie(z_counts, labels=z_counts.index, 
            autopct='%1.1f%%', colors=['#66b3ff', '#ff9999', '#99ff99'])
    plt.title('Z Distribution')
    plt.savefig(os.path.join(graficos_dir, 'z_distribution.png'))

def plot_correlation_heatmap(df):
    # Calculate correlation matrix for numerical columns
    correlation_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
    
    # Create a figure with appropriate size
    plt.figure(figsize=(10, 8))
    
    # Create heatmap using seaborn
    sns.heatmap(correlation_matrix, 
                annot=True,              # Show correlation values
                cmap='coolwarm',         # Color scheme from blue (negative) to red (positive)
                center=0,                # Center the colormap at 0
                fmt='.2f',              # Show 2 decimal places
                square=True,             # Make the plot square-shaped
                linewidths=0.5)         # Add lines between cells
    
    plt.title('Correlation Matrix of Numerical Variables')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(graficos_dir, 'correlation_heatmap.png'))

def plot_distribution(df):
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot histogram for x
    sns.histplot(data=df, x='x', bins=30, ax=ax1)
    ax1.set_title('Distribution of X')
    ax1.set_xlabel('X values')
    ax1.set_ylabel('Frequency')
    
    # Plot histogram for y
    sns.histplot(data=df, x='y', bins=30, ax=ax2)
    ax2.set_title('Distribution of Y')
    ax2.set_xlabel('Y values')
    ax2.set_ylabel('Frequency')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(graficos_dir, 'xy_distributions.png'))

def plot_boxplot(df):
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot boxplot for x grouped by z
    sns.boxplot(x='z', y='x', data=df, ax=ax1)
    ax1.set_title('Distribution of X by Z Category')
    ax1.set_xlabel('Z Category')
    ax1.set_ylabel('X values')
    
    # Plot boxplot for y grouped by z
    sns.boxplot(x='z', y='y', data=df, ax=ax2)
    ax2.set_title('Distribution of Y by Z Category')
    ax2.set_xlabel('Z Category')
    ax2.set_ylabel('Y values')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(graficos_dir, 'xy_boxplots_by_z.png'))

def plot_scatter(df):
    # Define color mapping dictionary using the same colors as z_distribution
    color_map = {-1: '#66b3ff', 0: '#ff9999', 1: '#99ff99'}
    
    # Create the scatter plot
    plt.figure(figsize=(10, 8))
    
    # Plot points for each z value with its corresponding color
    for z_value in sorted(df['z'].unique()):
        mask = df['z'] == z_value
        plt.scatter(df[mask]['x'], df[mask]['y'], 
                   c=color_map[z_value], 
                   label=f'Z = {z_value}',
                   alpha=0.6)
    
    plt.title('Scatter Plot of X vs Y by Z Category')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.legend()
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(graficos_dir, 'xy_scatter.png'))

# Create directory for saving graphs
current_dir = os.path.dirname(os.path.abspath(__file__))
graficos_dir = os.path.join(current_dir, 'graficos', 'exploracion')
os.makedirs(graficos_dir, exist_ok=True)

# Load and prepare the dataset
df = pd.read_excel('parcial/punto 1/Ej_1_A337_2025.xlsx')


# Execute all analysis functions
prints(df)
plot_z_distribution(df)
plot_correlation_heatmap(df)
plot_distribution(df)
plot_boxplot(df)
plot_scatter(df)


# Show all plots
plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')