import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


# Set up directories
current_dir = os.path.dirname(os.path.abspath(__file__))


# Load data
df = pd.read_csv('TP 7 Churn RNS/Iran Customer Churn.csv')
df = df.drop(['Status','Age'], axis=1)  # Remove Status variable


# Define categorical columns that should not be normalized
categorical_cols = ['Complains', 'Charge  Amount', 'Age Group', 'Tariff Plan', 'Churn']
numerical_cols = df.columns.difference(categorical_cols).tolist()


# Scale only the numerical columns
scaler = StandardScaler()
df_numerical_scaled = pd.DataFrame(
    scaler.fit_transform(df[numerical_cols]),
    columns=numerical_cols
)

# Combine the original categorical columns with the scaled numerical columns
df_scaled = pd.concat([df[categorical_cols], df_numerical_scaled], axis=1)

print(df_scaled.head())
print(df_scaled.describe())
print(df_scaled.info())

# Save the scaled data to a CSV file
df_scaled.to_excel('TP 7 Churn RNS/Iran Customer Churn_scaled.xlsx', index=False)