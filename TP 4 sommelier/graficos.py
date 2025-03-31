import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np



def plot_boxplots(combined_data):
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 12))
    axes1 = axes1.flatten()

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 12))
    axes2 = axes2.flatten()

    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 12))
    axes3 = axes3.flatten()

    numeric_columns = combined_data.drop('type', axis=1).columns
    num_vars = len(numeric_columns)
    first_third = num_vars // 3
    second_third = 2 * (num_vars // 3)

    for i, column in enumerate(numeric_columns[:first_third]):
        sns.boxplot(x='type', y=column, data=combined_data, ax=axes1[i])
        axes1[i].set_title(f'Distribution of {column}')
        axes1[i].set_xlabel('Wine Type')
        axes1[i].set_ylabel(column)

    for i, column in enumerate(numeric_columns[first_third:second_third]):
        sns.boxplot(x='type', y=column, data=combined_data, ax=axes2[i])
        axes2[i].set_title(f'Distribution of {column}')
        axes2[i].set_xlabel('Wine Type')
        axes2[i].set_ylabel(column)

    for i, column in enumerate(numeric_columns[second_third:]):
        if i < 4:
            sns.boxplot(x='type', y=column, data=combined_data, ax=axes3[i])
            axes3[i].set_title(f'Distribution of {column}')
            axes3[i].set_xlabel('Wine Type')
            axes3[i].set_ylabel(column)

    fig1.tight_layout(pad=3.0)
    fig1.subplots_adjust(top=0.9)

    fig2.tight_layout(pad=3.0)
    fig2.subplots_adjust(top=0.9)

    fig3.tight_layout(pad=3.0)
    fig3.subplots_adjust(top=0.9)
    
    return fig1, fig2, fig3


def plot_histograms(combined_data):
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 12))
    axes1 = axes1.flatten()

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 12))
    axes2 = axes2.flatten()

    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 12))
    axes3 = axes3.flatten()

    numeric_columns = combined_data.drop('type', axis=1).columns
    num_vars = len(numeric_columns)
    first_third = num_vars // 3
    second_third = 2 * (num_vars // 3)

    for i, column in enumerate(numeric_columns[:first_third]):
        sns.histplot(data=combined_data, x=column, hue='type', 
                     element='step', kde=True, common_norm=False, 
                     palette={'white': 'skyblue', 'red': 'salmon'}, 
                     ax=axes1[i])
        axes1[i].set_title(f'Distribution of {column}')
        axes1[i].set_xlabel(column)
        axes1[i].set_ylabel('Frequency')

    for i, column in enumerate(numeric_columns[first_third:second_third]):
        sns.histplot(data=combined_data, x=column, hue='type', 
                     element='step', kde=True, common_norm=False, 
                     palette={'white': 'skyblue', 'red': 'salmon'}, 
                     ax=axes2[i])
        axes2[i].set_title(f'Distribution of {column}')
        axes2[i].set_xlabel(column)
        axes2[i].set_ylabel('Frequency')

    for i, column in enumerate(numeric_columns[second_third:]):
        if i < 4:
            sns.histplot(data=combined_data, x=column, hue='type', 
                         element='step', kde=True, common_norm=False, 
                         palette={'white': 'skyblue', 'red': 'salmon'}, 
                         ax=axes3[i])
            axes3[i].set_title(f'Distribution of {column}')
            axes3[i].set_xlabel(column)
            axes3[i].set_ylabel('Frequency')

    fig1.tight_layout(pad=3.0)
    fig1.subplots_adjust(top=0.9)
    fig1.suptitle('Histograms of Wine Variables by Type (Part 1)', y=0.98, fontsize=16)

    fig2.tight_layout(pad=3.0)
    fig2.subplots_adjust(top=0.9)
    fig2.suptitle('Histograms of Wine Variables by Type (Part 2)', y=0.98, fontsize=16)

    fig3.tight_layout(pad=3.0)
    fig3.subplots_adjust(top=0.9)
    fig3.suptitle('Histograms of Wine Variables by Type (Part 3)', y=0.98, fontsize=16)
    
    return fig1, fig2, fig3


white_wine_data = pd.read_csv('TP 4 sommelier/winequality-white.csv', sep=';')
red_wine_data = pd.read_csv('TP 4 sommelier/winequality-red.csv', sep=';')


print("\nWhite Wine - First 5 rows:")
print(white_wine_data.head())
print("\nRed Wine - First 5 rows:")
print(red_wine_data.head())

print("\nWhite Wine - Statistical Summary:")
print(white_wine_data.describe())
print("\nRed Wine - Statistical Summary:")
print(red_wine_data.describe())

print("\nWhite Wine - Information:")
print(white_wine_data.info())
print("\nRed Wine - Information:")
print(red_wine_data.info())

white_wine_data['type'] = 'white'
red_wine_data['type'] = 'red'
combined_data = pd.concat([white_wine_data, red_wine_data], axis=0)
print(combined_data.head())

plot_boxplots(combined_data)

plot_histograms(combined_data)

plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')