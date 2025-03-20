import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np


def analyze_anova(combined_data):

    # Get numeric columns. xclude 'type' which is categorical and 'quality' because its another objective variable, not an inherent property of the wine
    numeric_columns = combined_data.drop(['type', 'quality'], axis=1).columns
    
    # Create a dataframe to store results
    results = []
    
    # Perform ANOVA for each variable
    for column in numeric_columns:
        # Use the existing white_wine_data and red_wine_data instead of creating new groups
        white_group = white_wine_data[column]
        red_group = red_wine_data[column]
        
        # Perform one-way ANOVA
        f_val, p_val = stats.f_oneway(white_group, red_group)
        
        # Calculate effect size (Eta-squared)
        # Total sum of squares
        total_ss = ((combined_data[column] - combined_data[column].mean())**2).sum()
        
        # Between-groups sum of squares
        white_mean = white_group.mean()
        red_mean = red_group.mean()
        grand_mean = combined_data[column].mean()
        
        between_ss = (((white_mean - grand_mean)**2) * len(white_group) + 
                      ((red_mean - grand_mean)**2) * len(red_group))
        
        # Eta-squared
        eta_squared = between_ss / total_ss
        
        # Store results
        results.append({
            'Variable': column,
            'F-statistic': f_val,
            'P-value': p_val,
            'Eta-squared': eta_squared,
            'Significant': p_val < 0.05
        })
    
    # Convert to DataFrame and sort by effect size
    result_df = pd.DataFrame(results)
    # Rename columns to avoid hyphen issues
    result_df = result_df.rename(columns={'Eta-squared': 'Eta_squared', 'P-value': 'P_value', 'F-statistic': 'F_statistic'})
    result_df = result_df.sort_values('Eta_squared', ascending=False)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Create bar plot of effect sizes
    ax = sns.barplot(x='Eta_squared', y='Variable', hue='Significant', data=result_df,
                    palette=['lightblue', 'darkblue'], legend=False)
    
    # Add significance markers
    for i, row in enumerate(result_df.itertuples()):
        if row.Significant:
            # Access using attribute name now that we've renamed the columns
            ax.text(row.Eta_squared + 0.01, i, '*', fontsize=16, color='black')
            # Add more stars for higher significance
            if row.P_value < 0.01:
                ax.text(row.Eta_squared + 0.03, i, '*', fontsize=16, color='black')
            if row.P_value < 0.001:
                ax.text(row.Eta_squared + 0.05, i, '*', fontsize=16, color='black')
    
    plt.title('Variable Importance for Distinguishing Wine Types (ANOVA)', fontsize=16)
    plt.xlabel('Effect Size (Eta-squared)', fontsize=12)
    plt.ylabel('Wine Properties', fontsize=12)
    plt.tight_layout()
    
    # Print the results table
    print("\nANOVA Analysis Results:")
    print(result_df[['Variable', 'F_statistic', 'P_value', 'Eta_squared', 'Significant']])
    
    return result_df, plt.gcf()


def train_knn_model(combined_data):
    # Get features and target
    X = combined_data.drop(['type', 'quality'], axis=1)  # Drop quality too if you don't want it as a feature
    y = combined_data['type']
    
    # Standardize features (important for KNN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Set up k-fold cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Test different k values
    k_values = range(1, 51)
    mean_accuracies = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_scaled, y, cv=kf, scoring='accuracy')
        mean_accuracies.append(scores.mean())
    
    # Find best k
    best_k = k_values[np.argmax(mean_accuracies)]
    best_accuracy = max(mean_accuracies)
    
    # Plot k vs accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, mean_accuracies, 'o-')
    plt.axvline(x=best_k, color='r', linestyle='--')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title(f'KNN: Accuracy vs k (Best k = {best_k}, Accuracy = {best_accuracy:.4f})')
    plt.grid(True)
    
    # Train final model with best k
    final_model = KNeighborsClassifier(n_neighbors=best_k)
    final_model.fit(X_scaled, y)
    
    return final_model, scaler, best_k, best_accuracy


def plot_full_model_decision_boundary(combined_data, knn_model, scaler, anova_results):
    """
    Plot the decision boundary for the full KNN model using the two most important features.
    """
    # Get the two most important features based on ANOVA results
    top_features = anova_results['Variable'].tolist()[:2]
    
    print(f"Plotting full model decision boundary using features: {top_features}")
    
    # Get the features and target
    X = combined_data[top_features]
    y = combined_data['type']
    
    # Create a meshgrid for plotting the decision boundary
    x_min, x_max = X[top_features[0]].min() - 0.1, X[top_features[0]].max() + 0.1
    y_min, y_max = X[top_features[1]].min() - (X[top_features[1]].max() - X[top_features[1]].min()) * 0.1, X[top_features[1]].max() + (X[top_features[1]].max() - X[top_features[1]].min()) * 0.1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min) / 100),
                         np.arange(y_min, y_max, (y_max - y_min) / 100))
    
    # For prediction, we need to include all features used in the model
    all_features = combined_data.drop(['type', 'quality'], axis=1).columns.tolist()
    
    # Create a dataframe with the mean values for all features
    grid = pd.DataFrame(columns=all_features)
    
    # Create a meshgrid as a DataFrame with proper column names
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    total_points = mesh_points.shape[0]
    
    # Initialize with means
    for feature in all_features:
        grid[feature] = np.repeat(combined_data[feature].mean(), total_points)
    
    # Set the values for our two visualization features
    grid[top_features[0]] = mesh_points[:, 0]
    grid[top_features[1]] = mesh_points[:, 1]
    
    # Scale the grid points
    grid_scaled = scaler.transform(grid)
    
    # Predict classes
    Z = knn_model.predict(grid_scaled)
    
    # Convert to numeric values (0 for white, 1 for red)
    Z_numeric = np.array([1 if z == 'red' else 0 for z in Z])
    Z_numeric = Z_numeric.reshape(xx.shape)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot the decision boundary with numeric values
    plt.contourf(xx, yy, Z_numeric, alpha=0.3, 
                 cmap=plt.cm.RdBu_r)  # Note: RdBu_r to make red=1, blue=0
    
    # Plot the data points
    scatter = plt.scatter(X[top_features[0]], X[top_features[1]], 
                c=[0 if t == 'white' else 1 for t in y], 
                cmap=plt.cm.RdBu_r, edgecolors='k', s=40, alpha=0.7)
    
    plt.xlabel(top_features[0], fontsize=12)
    plt.ylabel(top_features[1], fontsize=12)
    plt.title(f'Full KNN Model Decision Boundary (all features, accuracy={best_accuracy:.4f})', fontsize=16)
    
    # Add color legend
    labels = ['White Wine', 'Red Wine']
    plt.legend(handles=scatter.legend_elements()[0], labels=labels)
    
    plt.tight_layout()
    return plt.gcf()


def plot_simple_knn_decision_boundary(combined_data, knn_model, scaler):
    # Get the features and target
    features = ['volatile acidity', 'total sulfur dioxide']
    X = combined_data[features]
    y = combined_data['type']
    
    # Create a meshgrid for plotting the decision boundary
    # Get min/max values for both features
    va_min, va_max = X['volatile acidity'].min() - 0.1, X['volatile acidity'].max() + 0.1
    tsd_min, tsd_max = X['total sulfur dioxide'].min() - 10, X['total sulfur dioxide'].max() + 10
    
    # Create the grid points - total sulfur dioxide for X-axis, volatile acidity for Y-axis
    tsd_grid = np.arange(tsd_min, tsd_max, 1)
    va_grid = np.arange(va_min, va_max, 0.01)
    xx, yy = np.meshgrid(tsd_grid, va_grid)  # xx is total sulfur dioxide, yy is volatile acidity
    
    # Create a dataframe with proper column names - in same order as training
    # Need to reshape the data points for prediction
    grid = pd.DataFrame({
        'volatile acidity': yy.ravel(),        # yy is volatile acidity
        'total sulfur dioxide': xx.ravel()     # xx is total sulfur dioxide
    })
    
    # Scale the meshgrid points
    grid_scaled = scaler.transform(grid)
    
    # Predict classes for the meshgrid
    Z = knn_model.predict(grid_scaled)
    # Convert to numeric values (0 for white, 1 for red)
    Z_numeric = np.array([1 if z == 'red' else 0 for z in Z])
    Z_numeric = Z_numeric.reshape(xx.shape)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot the decision boundary - xx and yy are already in the correct order for contourf
    plt.contourf(xx, yy, Z_numeric, alpha=0.3, cmap=plt.cm.RdBu_r)
    
    # Plot the data points
    scatter = plt.scatter(
        X['total sulfur dioxide'],   # x-axis is total sulfur dioxide
        X['volatile acidity'],       # y-axis is volatile acidity
        c=[0 if t == 'white' else 1 for t in y], 
        cmap=plt.cm.RdBu_r, edgecolors='k', s=40, alpha=0.7
    )
    
    # Axis labels
    plt.xlabel('Total Sulfur Dioxide', fontsize=12)
    plt.ylabel('Volatile Acidity', fontsize=12)
    plt.title(f'Simple KNN Model Decision Boundary (2 features, accuracy={simple_best_accuracy:.4f})', fontsize=16)
    
    # Add color legend
    labels = ['White Wine', 'Red Wine']
    plt.legend(handles=scatter.legend_elements()[0], labels=labels)
    
    plt.tight_layout()
    return plt.gcf()


def train_simple_knn_model(combined_data):
    # Get only the two features we want
    X = combined_data[['volatile acidity', 'total sulfur dioxide']]
    y = combined_data['type']
    
    # Standardize features (important for KNN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Set up k-fold cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Test different k values
    k_values = range(1, 51)
    mean_accuracies = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_scaled, y, cv=kf, scoring='accuracy')
        mean_accuracies.append(scores.mean())
    
    # Find best k
    best_k = k_values[np.argmax(mean_accuracies)]
    best_accuracy = max(mean_accuracies)
    
    # Plot k vs accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, mean_accuracies, 'o-')
    plt.axvline(x=best_k, color='r', linestyle='--')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title(f'Simple KNN (2 features): Accuracy vs k (Best k = {best_k}, Accuracy = {best_accuracy:.4f})')
    plt.grid(True)
    
    # Train final model with best k
    final_model = KNeighborsClassifier(n_neighbors=best_k)
    final_model.fit(X_scaled, y)
    
    return final_model, scaler, best_k, best_accuracy


def perform_feature_sensitivity_analysis(combined_data, anova_results):
    """
    Perform sensitivity analysis by iteratively removing features
    based on ANOVA results to find the best performing model.
    """
    print("\nStarting Feature Sensitivity Analysis...")
    
    # Get features and target
    X = combined_data.drop(['type', 'quality'], axis=1)
    y = combined_data['type']
    
    # Get all feature names
    all_features = X.columns.tolist()
    
    # Create a list of features ordered by their importance (least important first)
    # This is the order we'll try removing them
    features_by_importance = anova_results['Variable'].tolist()
    features_by_importance.reverse()  # Reverse to get least important first
    
    # Initialize tracking variables
    best_accuracy = 0
    best_k = 0
    best_features = all_features.copy()
    current_features = all_features.copy()
    
    # Initialize results tracking
    results = []
    
    # First, evaluate the model with all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Set up k-fold cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Find optimal k for all features
    k_values = range(1, 30)
    mean_accuracies = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_scaled, y, cv=kf, scoring='accuracy')
        mean_accuracies.append(scores.mean())
    
    # Record initial model with all features
    initial_best_k = k_values[np.argmax(mean_accuracies)]
    initial_accuracy = max(mean_accuracies)
    
    best_accuracy = initial_accuracy
    best_k = initial_best_k
    
    results.append({
        'Features Removed': 'None',
        'Features Used': len(current_features),
        'Best k': initial_best_k,
        'Accuracy': initial_accuracy
    })
    
    print(f"Initial model (all {len(current_features)} features): Accuracy = {initial_accuracy:.4f}, k = {initial_best_k}")
    
    # Now start removing features one by one
    for feature_to_remove in features_by_importance:
        # Skip if the feature is not in our current list
        if feature_to_remove not in current_features:
            continue
            
        # Remove the feature
        current_features.remove(feature_to_remove)
        
        # If we've removed all features, break
        if len(current_features) == 0:
            break
            
        # Prepare the data with current features
        X_reduced = X[current_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reduced)
        
        # Find optimal k for this feature set
        mean_accuracies = []
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X_scaled, y, cv=kf, scoring='accuracy')
            mean_accuracies.append(scores.mean())
        
        current_best_k = k_values[np.argmax(mean_accuracies)]
        current_accuracy = max(mean_accuracies)
        
        # Record results
        results.append({
            'Features Removed': feature_to_remove,
            'Features Used': len(current_features),
            'Best k': current_best_k,
            'Accuracy': current_accuracy
        })
        
        print(f"Removed '{feature_to_remove}': Accuracy = {current_accuracy:.4f}, k = {current_best_k}")
        
        # Update best model if this one is better
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_k = current_best_k
            best_features = current_features.copy()
            print(f"  --> New best model found!")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Train the best model
    X_best = X[best_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_best)
    
    best_model = KNeighborsClassifier(n_neighbors=best_k)
    best_model.fit(X_scaled, y)
    
    print("\nBest Model Summary:")
    print(f"Features: {best_features}")
    print(f"Number of features: {len(best_features)}")
    print(f"Optimal k: {best_k}")
    print(f"Cross-validated accuracy: {best_accuracy:.4f}")
    
    # Create visualization of accuracy vs features removed
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(results)), results_df['Accuracy'], 'o-', linewidth=2)
    plt.xlabel('Iteration (Features Removed)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('KNN Model Accuracy vs. Features Removed', fontsize=16)
    plt.grid(True)
    
    # Add annotations for removed features
    for i, row in enumerate(results_df.itertuples()):
        if i > 0:  # Skip the first iteration (no features removed)
            plt.annotate(row._2,  # 'Features Removed' column
                        xy=(i, row.Accuracy),
                        xytext=(5, 0),
                        textcoords='offset points',
                        rotation=45,
                        fontsize=8)
    
    # Highlight the best model
    best_idx = results_df['Accuracy'].idxmax()
    plt.plot(best_idx, results_df.loc[best_idx, 'Accuracy'], 'ro', markersize=10)
    plt.annotate('Best Model',
                xy=(best_idx, results_df.loc[best_idx, 'Accuracy']),
                xytext=(10, -20),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->'),
                fontsize=12)
    
    plt.tight_layout()
    
    return best_model, scaler, best_features, best_k, best_accuracy, plt.gcf()


def plot_best_model_k_accuracy(combined_data, best_features):
    """
    Plot the accuracy vs k curve for the best model with optimal feature set.
    """
    # Get the data with only the best features
    X = combined_data[best_features]
    y = combined_data['type']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Set up cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Test different k values
    k_values = range(1, 51)
    mean_accuracies = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_scaled, y, cv=kf, scoring='accuracy')
        mean_accuracies.append(scores.mean())
    
    # Find best k
    best_k = k_values[np.argmax(mean_accuracies)]
    best_accuracy = max(mean_accuracies)
    
    # Plot k vs accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, mean_accuracies, 'o-')
    plt.axvline(x=best_k, color='r', linestyle='--')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title(f'Best KNN Model ({len(best_features)} features): Accuracy vs k (Best k = {best_k}, Accuracy = {best_accuracy:.4f})')
    plt.grid(True)
    
    return plt.gcf(), best_k, best_accuracy


def plot_best_model_decision_boundary(combined_data, best_model, best_scaler, best_features, anova_results):
    """
    Plot the decision boundary for the best model using the two most important features.
    """
    # If we have more than 2 features, we need to select the 2 most important ones for visualization
    if len(best_features) > 2:
        # Get the two most important features from our best features based on ANOVA results
        top_features = [feat for feat in anova_results['Variable'] if feat in best_features][:2]
    else:
        top_features = best_features
    
    print(f"Plotting decision boundary using features: {top_features}")
    
    # Get the features and target
    X = combined_data[top_features]
    y = combined_data['type']
    
    # Create a meshgrid for plotting the decision boundary
    x_min, x_max = X[top_features[0]].min() - 0.1, X[top_features[0]].max() + 0.1
    y_min, y_max = X[top_features[1]].min() - (X[top_features[1]].max() - X[top_features[1]].min()) * 0.1, X[top_features[1]].max() + (X[top_features[1]].max() - X[top_features[1]].min()) * 0.1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min) / 100),
                         np.arange(y_min, y_max, (y_max - y_min) / 100))
    
    # For prediction, we need data in the format of all best features
    if len(best_features) > 2:
        # Create a dataframe with the mean values for all features
        grid = pd.DataFrame(columns=best_features)
        
        # Initialize with means
        total_points = xx.ravel().shape[0]
        for feature in best_features:
            grid[feature] = np.repeat(combined_data[feature].mean(), total_points)
        
        # Set the values for our two visualization features
        grid[top_features[0]] = xx.ravel()
        grid[top_features[1]] = yy.ravel()
        
        # Scale the grid points
        grid_scaled = best_scaler.transform(grid)
        
        # Predict classes
        Z = best_model.predict(grid_scaled)
    else:
        # If we're just using 2 features, it's simpler
        grid = pd.DataFrame(columns=best_features)
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        grid[best_features[0]] = mesh_points[:, 0]
        grid[best_features[1]] = mesh_points[:, 1]
        
        grid_scaled = best_scaler.transform(grid)
        Z = best_model.predict(grid_scaled)
    
    # Convert to numeric values (0 for white, 1 for red)
    Z_numeric = np.array([1 if z == 'red' else 0 for z in Z])
    Z_numeric = Z_numeric.reshape(xx.shape)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot the decision boundary with numeric values
    plt.contourf(xx, yy, Z_numeric, alpha=0.3, 
                 cmap=plt.cm.RdBu_r)  # Note: RdBu_r to make red=1, blue=0
    
    # Plot the data points
    scatter = plt.scatter(X[top_features[0]], X[top_features[1]], 
                c=[0 if t == 'white' else 1 for t in y], 
                cmap=plt.cm.RdBu_r, edgecolors='k', s=40, alpha=0.7)
    
    plt.xlabel(top_features[0], fontsize=12)
    plt.ylabel(top_features[1], fontsize=12)
    plt.title(f'Best KNN Model Decision Boundary ({len(best_features)} features, accuracy={best_accuracy:.4f})', fontsize=16)
    
    # Add color legend
    labels = ['White Wine', 'Red Wine']
    plt.legend(handles=scatter.legend_elements()[0], labels=labels)
    
    plt.tight_layout()
    return plt.gcf()


# Load the datasets
white_wine_data = pd.read_csv('TP 4 sommelier/winequality-white.csv', sep=';')
red_wine_data = pd.read_csv('TP 4 sommelier/winequality-red.csv', sep=';')

#agregar columna a los dataframes para indicar el tipo de vino
white_wine_data['type'] = 'white'
red_wine_data['type'] = 'red'

#juntar los datos de los dos datasets
combined_data = pd.concat([white_wine_data, red_wine_data], axis=0)



# Perform one-way ANOVA tests to analyze the relationship between each continuous variable and wine type.
anova_results, anova_fig = analyze_anova(combined_data)

print('arranca knn')
# Train the model and visualize
knn_model, scaler, best_k, best_accuracy = train_knn_model(combined_data)

print('Creating decision boundary for full model...')
full_decision_boundary = plot_full_model_decision_boundary(combined_data, knn_model, scaler, anova_results)

print('arranca knn simple')
# Train the simpler model with just 2 features
simple_knn_model, simple_scaler, simple_best_k, simple_best_accuracy = train_simple_knn_model(combined_data)

# Plot the decision boundary
decision_boundary_plot = plot_simple_knn_decision_boundary(combined_data, simple_knn_model, simple_scaler)

print('Starting sensitivity analysis to find the best model...')
best_model, best_scaler, best_features, best_k, best_accuracy, sensitivity_fig = perform_feature_sensitivity_analysis(combined_data, anova_results)

print('Plotting accuracy vs k for the best model...')
best_k_plot, confirmed_best_k, confirmed_best_accuracy = plot_best_model_k_accuracy(combined_data, best_features)

print('Creating decision boundary for the best model...')
best_decision_boundary = plot_best_model_decision_boundary(combined_data, best_model, best_scaler, best_features, anova_results)

plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')