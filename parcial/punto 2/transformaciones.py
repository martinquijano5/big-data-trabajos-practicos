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

def float_to_int():
    print(df.info())
    
    for col in df.columns:
        df[col] = df[col].astype('int64')

def bmi_category(input_df):
    local_df = input_df.copy()
    bins = [0, 18.5, 25, 30, 35, 40, float('inf')]
    labels = ['Underweight', 'Normal', 'Overweight', 'Obese_I', 'Obese_II', 'Obese_III']
    original_cols = set(local_df.columns)

    local_df['BMI_Category'] = pd.cut(local_df['BMI'], bins=bins, labels=labels, right=False)

    print("Value counts for BMI_Category:")
    print(local_df['BMI_Category'].value_counts())

    local_df = pd.get_dummies(local_df, columns=['BMI_Category'], drop_first=True)

    # Identify new dummy columns
    new_dummy_cols = list(set(local_df.columns) - original_cols - {'BMI_Category'}) # Find columns added by get_dummies
     # Convert boolean dummies to int
    for col in new_dummy_cols:
        if local_df[col].dtype == 'bool':
             local_df[col] = local_df[col].astype(int)


    local_df = local_df.drop('BMI', axis=1)

    print("\nDataFrame head after BMI transformation (int dummies):")
    print(local_df.head())
    return local_df

def one_hot_encoding(input_df):
    local_df = input_df.copy()
    categorical_cols_to_encode = ['Age', 'Education', 'Income', 'GenHlth']
    original_cols = set(local_df.columns)

    cols_to_encode_present = [col for col in categorical_cols_to_encode if col in local_df.columns]
    if not cols_to_encode_present:
        print("Warning: None of the specified columns for OHE found in DataFrame.")
        return local_df
    if len(cols_to_encode_present) < len(categorical_cols_to_encode):
        missing_cols = set(categorical_cols_to_encode) - set(cols_to_encode_present)
        print(f"Warning: Columns not found for OHE: {missing_cols}")
        # Filter out missing columns from the list to encode
        cols_to_encode_present = [col for col in categorical_cols_to_encode if col in cols_to_encode_present]
        if not cols_to_encode_present: # If none are left
             return local_df


    print("\nColumns before OHE:", local_df.columns)
    print("Shape before OHE:", local_df.shape)

    local_df = pd.get_dummies(local_df, columns=cols_to_encode_present, drop_first=True)

    # Identify new dummy columns
    new_dummy_cols = list(set(local_df.columns) - original_cols)
    # Convert boolean dummies to int
    for col in new_dummy_cols:
        if local_df[col].dtype == 'bool':
             local_df[col] = local_df[col].astype(int)


    print("\nColumns after OHE (int dummies):", local_df.columns)
    print("Shape after OHE:", local_df.shape)
    print("\nDataFrame head after OHE (int dummies):")
    print(local_df.head())
    return local_df

def bin_health_days(input_df):
    local_df = input_df.copy()
    cols_to_bin = ['MentHlth', 'PhysHlth']
    original_cols = set(local_df.columns)

    # Check if columns exist
    cols_present = [col for col in cols_to_bin if col in local_df.columns]
    if not cols_present:
        print("Warning: MentHlth or PhysHlth columns not found for binning.")
        return local_df

    # Define bins and NEW numerical labels
    bins = [-float('inf'), 0, 5, 10, 15, 20, 25, 30]
    # Label 1 = 0 days, 2 = 1-5 days, ..., 7 = 26-30 days
    num_labels = [1, 2, 3, 4, 5, 6, 7]
    label_map = {1: '0 days', 2: '1-5 days', 3: '6-10 days', 4: '11-15 days', 5: '16-20 days', 6: '21-25 days', 7: '26-30 days'}


    print("\nBinning MentHlth and PhysHlth using numerical labels...")
    temp_binned_cols = [] # To store the temporary numerical column names before OHE
    col_prefixes = [] # To store prefixes for get_dummies
    for col in cols_present:
        binned_col_name = f'{col}_BinNum' # Create intermediate col with numerical labels
        local_df[binned_col_name] = pd.cut(local_df[col], bins=bins, labels=num_labels, right=True, include_lowest=True) # include_lowest ensures 0 is captured
        print(f"\nValue counts for {binned_col_name} (Numerical Labels):")
        # Print value counts with original meaning for clarity
        counts = local_df[binned_col_name].value_counts().sort_index()
        counts.index = counts.index.map(label_map)
        print(counts)

        temp_binned_cols.append(binned_col_name)
        col_prefixes.append(col) # Use original column name as prefix

    # One-hot encode the new numerical binned columns
    if temp_binned_cols:
        # Apply get_dummies to the numerical bin columns
        # prefix will create names like MentHlth_2, MentHlth_3 etc. (after drop_first)
        local_df = pd.get_dummies(local_df, columns=temp_binned_cols, prefix=col_prefixes, prefix_sep='_Bin_', drop_first=True)

        # Identify new dummy columns (names like MentHlth_Bin_2, PhysHlth_Bin_5 etc)
        # Need to exclude original columns AND the intermediate _BinNum columns (which were removed by get_dummies)
        new_dummy_cols = list(set(local_df.columns) - original_cols)

        # Convert boolean dummies to int
        for col in new_dummy_cols:
             # Check if the column name matches the expected pattern for new dummies
             is_new_dummy = any(col.startswith(f"{prefix}_Bin_") for prefix in col_prefixes)
             if is_new_dummy and local_df[col].dtype == 'bool':
                 local_df[col] = local_df[col].astype(int)


        # Drop ONLY the original columns (MentHlth, PhysHlth)
        # The intermediate _BinNum columns were removed by get_dummies
        local_df = local_df.drop(columns=cols_present)
        print("\nDataFrame head after binning and OHE (numerical bin names):")
        print(local_df.head())
    else:
         print("No columns were binned.")


    return local_df


current_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv('punto 2/diabetes_binary_health_indicators_BRFSS2015.csv')

float_to_int()

# Apply BMI categorization
df = bmi_category(df)

# Apply One-Hot Encoding for other categoricals
df = one_hot_encoding(df)

# Apply Binning for health days
df = bin_health_days(df)

print("\nFinal DataFrame info after all transformations:")
df.info()


df.to_csv('punto 2/diabetes_binary_health_indicators_BRFSS2015_transformed.csv', index=False)