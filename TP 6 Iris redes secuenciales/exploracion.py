import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from itertools import combinations

def prints(df):
    print("\nIris - First 5 rows:")
    print(df.head())

    print("\nIris - Statistical Summary:")
    print(df.describe())

    print("\nIris - Data Types:")
    print(df.dtypes)

    print("\nIris - Missing Values:")
    print(df.isnull().sum())

def proportions(df):
    # Calculate the proportions of each species
    species_counts = df['Species'].value_counts()
    species_proportions = species_counts / species_counts.sum() * 100

    # Create a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(species_proportions, labels=species_proportions.index, autopct='%1.1f%%', startangle=90)
    plt.title('Proportions of Iris Species')
    plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
    plt.tight_layout()
    plt.savefig(os.path.join(graficos_dir, 'proportions.png'))

def boxplots(df):
    # Create subplots for boxplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Boxplots by Species')

    # Boxplots of each variable by species
    species = df['Species'].unique()
    for i, (col, ax) in enumerate(zip(['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'], axes.flat)):
        data = [df[df['Species'] == s][col] for s in species]
        ax.boxplot(data, labels=species)
        ax.set_title(col)
        ax.set_ylabel('cm')

    plt.tight_layout()
    plt.savefig(os.path.join(graficos_dir, 'boxplots.png'))

def histograms(df):
    # Create subplots for histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Histograms of Iris Features')

    # Histograms of each variable
    for i, (col, ax) in enumerate(zip(['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'], axes.flat)):
        ax.hist(df[col], bins=20, density=True, alpha=0.7)
        ax.set_title(col)
        ax.set_xlabel('cm')
        ax.set_ylabel('Density')
        
        # Add KDE
        kde = stats.gaussian_kde(df[col])
        x_range = np.linspace(df[col].min(), df[col].max(), 100)
        ax.plot(x_range, kde(x_range), 'r-', lw=2)

    plt.tight_layout()
    plt.savefig(os.path.join(graficos_dir, 'histograms.png'))

def scatter(df):
    # All possible combinations of features
    feature_combinations = [
        ('SepalLengthCm', 'SepalWidthCm'),
        ('SepalLengthCm', 'PetalLengthCm'),
        ('SepalLengthCm', 'PetalWidthCm'),
        ('SepalWidthCm', 'PetalLengthCm'),
        ('SepalWidthCm', 'PetalWidthCm'),
        ('PetalLengthCm', 'PetalWidthCm')
    ]
    
    # Create three separate figures, each with 2 subplots
    species_colors = {'Iris-setosa': 'blue', 'Iris-versicolor': 'orange', 'Iris-virginica': 'green'}
    
    for fig_num in range(3):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f'Scatter Plots of Iris Features (Set {fig_num+1})')
        
        # Get the two combinations for this figure
        start_idx = fig_num * 2
        current_combinations = feature_combinations[start_idx:start_idx+2]
        
        for i, ((col1, col2), ax) in enumerate(zip(current_combinations, axes.flatten())):
            for species, color in species_colors.items():
                ax.scatter(df[df['Species'] == species][col1], df[df['Species'] == species][col2], 
                          alpha=0.7, label=species, color=color)
            ax.set_title(f'{col1} vs {col2}')
            ax.set_xlabel(col1)
            ax.set_ylabel(col2)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(graficos_dir, f'scatter_{fig_num+1}.png'))

def plot_3d(df):
    # Get feature names (excluding 'Id' and target column)
    feature_cols = [col for col in df.columns if col not in ['Id', 'Species']]
    
    # Generate all possible 3-variable combinations
    feature_combinations = list(combinations(feature_cols, 3))
    print(f"Combinaciones 3D de features: {feature_combinations}")
    
    # Create a figure for each combination
    for combinacion in feature_combinations:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract features for this combination
        features = list(combinacion)
        
        # Plot each species with different color
        for species, color in species_colors.items():
            # Filter data for this species
            species_data = df[df['Species'] == species]
            
            # Plot the 3D scatter for this species
            ax.scatter(
                species_data[features[0]],
                species_data[features[1]],
                species_data[features[2]],
                c=color,
                label=species,
                s=50,
                edgecolor='black',
                alpha=0.7
            )
        
        # Set labels and title
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel(features[2])
        ax.set_title(f'3D Scatter Plot: {features[0]} vs {features[1]} vs {features[2]}')
        
        # Add legend
        ax.legend()
        
        # Adjust view angle
        ax.view_init(elev=30, azim=45)
        
        # Save figure
        combinacion_name = f"{features[0]}_{features[1]}_{features[2]}"
        plt.savefig(os.path.join(graficos_dir, f'3d_scatter_{combinacion_name}.png'), dpi=300)
        plt.tight_layout()

# Define species colors for consistent visualization
species_colors = {'Iris-setosa': 'blue', 'Iris-versicolor': 'orange', 'Iris-virginica': 'green'}

current_dir = os.path.dirname(os.path.abspath(__file__))
# Create graficos directory if it doesn't exist
graficos_dir = os.path.join(current_dir, 'graficos', 'exploracion')
os.makedirs(graficos_dir, exist_ok=True)

iris_df = pd.read_csv(os.path.join(current_dir, 'iris.csv'))

prints(iris_df)
proportions(iris_df)
boxplots(iris_df)
histograms(iris_df)
scatter(iris_df)
plot_3d(iris_df)



# Standardize the data (excluding Id and Species columns)
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(iris_df[features])
scaled_df = pd.DataFrame(scaled_features, columns=features)

# Print the standardized data
print("\nComplete Standardized Iris Data:")
pd.set_option('display.max_rows', None)  # Show all rows
print(scaled_df)
pd.reset_option('display.max_rows')  # Reset to default setting

# Save scaled data to CSV with explicit comma separator
scaled_csv_path = os.path.join(graficos_dir, 'scaled_iris_data.csv')
scaled_df.to_csv(scaled_csv_path, sep=',', index=False)
print(f"\nScaled data saved to: {scaled_csv_path}")




plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')