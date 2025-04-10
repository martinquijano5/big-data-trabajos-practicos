import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
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

# Set style for better visualizations
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the dataset
df = pd.read_csv('TP 2 Churn Iran customer/Iran Customer Churn.csv')

# Remove Status variable from the analysis
df = df.drop('Status', axis=1)

# Basic data exploration
print("Head of the dataset:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nDescriptive statistics:")
print(df.describe())


# Check class distribution (churn vs non-churn)
print("\nChurn distribution:")
churn_counts = df['Churn'].value_counts()
print(churn_counts)
print(f"Churn rate: {churn_counts[1] / len(df) * 100:.2f}%")


# Create directory for saving graphs
current_dir = os.path.dirname(os.path.abspath(__file__))
graficos_dir = os.path.join(current_dir, 'graficos', 'exploracion')
os.makedirs(graficos_dir, exist_ok=True)

# Visualizations
# 1. Churn Distribution - Pie Chart
plt.figure(figsize=(10, 6))
plt.pie(churn_counts, labels=['Non-Churn', 'Churn'], autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
plt.title('Churn Distribution')
plt.savefig(os.path.join(graficos_dir, 'churn_distribution.png'))

# 2. Correlation Heatmap
plt.figure(figsize=(14, 10))
correlation = df.corr()
mask = np.triu(correlation)
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5, mask=mask)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, 'correlation_heatmap.png'))

# 3. Numerical Features Distribution by Churn
numerical_features = [
    'Call  Failure', 'Seconds of Use', 'Frequency of use', 'Frequency of SMS', 
    'Distinct Called Numbers', 'Subscription  Length', 'Charge  Amount', 'Age Group',
    'Customer Value'  
]

# Split features into groups of 4
feature_groups = [numerical_features[i:i+4] for i in range(0, len(numerical_features), 4)]

# Create a separate figure for each group
for group_idx, feature_group in enumerate(feature_groups):
    if len(feature_group) < 4:
        n_features = len(feature_group)
        fig, axes = plt.subplots(1, n_features, figsize=(6*n_features, 5))
        if n_features == 1:
            axes = [axes]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
    
    for i, feature in enumerate(feature_group):
        sns.boxplot(x='Churn', y=feature, data=df, ax=axes[i])
        axes[i].set_title(f'{feature} by Churn Status', pad=20)
        axes[i].set_xlabel('Churn (0=No, 1=Yes)')
        axes[i].tick_params(axis='both', which='major', labelsize=10)
        
        # Rotate y-axis labels if they're too long
        if max([len(str(x)) for x in axes[i].get_yticks()]) > 4:
            axes[i].tick_params(axis='y', rotation=45)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(graficos_dir, f'numerical_features_group_{group_idx+1}.png'), 
                bbox_inches='tight', 
                dpi=300)

# 4. Categorical Features Analysis
fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # Changed from 1,3 to 1,2
categorical_features = ['Complains', 'Tariff Plan']  # Removed 'Status'

for i, feature in enumerate(categorical_features):
    # Create a crosstab
    ct = pd.crosstab(df[feature], df['Churn'])
    # Calculate percentages
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    
    # Plot
    ct_pct.plot(kind='bar', stacked=True, ax=axes[i], color=['#66b3ff', '#ff9999'])
    axes[i].set_title(f'Churn Rate by {feature}')
    axes[i].set_ylabel('Percentage')
    axes[i].set_xlabel(feature)
    
    # Add percentage labels
    for c in axes[i].containers:
        axes[i].bar_label(c, label_type='center', fmt='%.1f%%')

plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, 'categorical_features.png'))

# Add the same interactive closing mechanism
plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')