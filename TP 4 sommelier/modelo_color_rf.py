import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from sklearn.metrics import confusion_matrix, accuracy_score

# Change working directory to script location - add this near the top with other imports
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def analyze_anova(combined_data):

    # Get numeric columns. xclude 'type' which is categorical and 'quality' because its another objective variable, not an inherent property of the wine
    numeric_columns = combined_data.drop(['type', 'quality'], axis=1).columns
    
    # Create a copy of the data for standardization
    data_for_analysis = combined_data.copy()
    
    # Standardize the numeric columns
    scaler = StandardScaler()
    data_for_analysis[numeric_columns] = scaler.fit_transform(data_for_analysis[numeric_columns])
    
    # Split the standardized data by wine type
    white_data_std = data_for_analysis[data_for_analysis['type'] == 'white']
    red_data_std = data_for_analysis[data_for_analysis['type'] == 'red']
    
    # Create a dataframe to store results
    results = []
    
    # Perform ANOVA for each variable
    for column in numeric_columns:
        # Use the standardized data
        white_group = white_data_std[column]
        red_group = red_data_std[column]
        
        # Perform one-way ANOVA
        f_val, p_val = stats.f_oneway(white_group, red_group)
        
        # Calculate effect size (Eta-squared)
        # Total sum of squares for standardized data
        all_values = data_for_analysis[column]
        grand_mean = all_values.mean()
        total_ss = ((all_values - grand_mean)**2).sum()
        
        # Between-groups sum of squares
        white_mean = white_group.mean()
        red_mean = red_group.mean()
        
        between_ss = (len(white_group) * (white_mean - grand_mean)**2 + 
                      len(red_group) * (red_mean - grand_mean)**2)
        
        # Eta-squared - should be between 0 and 1
        eta_squared = between_ss / total_ss if total_ss != 0 else 0
        
        # Validation check
        if eta_squared > 1 or eta_squared < 0:
            print(f"Warning: Invalid eta_squared value for {column}: {eta_squared}")
            eta_squared = max(0, min(1, eta_squared))  # Clip to valid range
        
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
    
    # Replace the stars with the actual values
    for i, row in enumerate(result_df.itertuples()):
        # Format the value to show only 3 decimal places
        value_text = f"{row.Eta_squared:.3f}"
        # Position the text at the end of each bar
        ax.text(row.Eta_squared + 0.01, i, value_text, fontsize=10, va='center')
    
    # Ensure x-axis limits are appropriate
    max_value = result_df['Eta_squared'].max()
    plt.xlim(0, min(1, max_value * 1.2))  # Add some padding, but don't exceed 1
    
    plt.title('Variable Importance for Distinguishing Wine Types (ANOVA on Standardized Data)', fontsize=16)
    plt.xlabel('Effect Size (Eta-squared)', fontsize=12)
    plt.ylabel('Wine Properties', fontsize=12)
    plt.tight_layout()
    
    # Print the results table
    print("\nANOVA Analysis Results (on Standardized Data):")
    print(result_df[['Variable', 'F_statistic', 'P_value', 'Eta_squared', 'Significant']])
    
    return result_df, plt.gcf()


def train_rf_model(combined_data):
    # Get features and target
    X = combined_data.drop(['type', 'quality'], axis=1)
    y = combined_data['type']
    
    # Standardize features (though RF is less sensitive to scaling than KNN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Set up k-fold cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Define parameter grid for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create base model
    rf = RandomForestClassifier(random_state=42)
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=kf,
        scoring='accuracy',
        n_jobs=-1,  # Use all available cores
        verbose=1
    )
    
    # Perform grid search
    grid_search.fit(X_scaled, y)
    
    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_accuracy = grid_search.best_score_
    
    print("\nRandom Forest Grid Search Results:")
    print(f"Best Parameters: {best_params}")
    print(f"Best Cross-Validated Accuracy: {best_accuracy:.4f}")
    
    # Plot feature importance from the best model
    plt.figure(figsize=(12, 8))
    
    # Get feature importance
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X.columns
    
    # Create a DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': [features[i] for i in indices],
        'Importance': [importances[i] for i in indices]
    })
    
    importance_df['Significant'] = importance_df['Importance'] > np.mean(importances)
    
    # Plot feature importance with light blue color
    ax = sns.barplot(x='Importance', y='Feature', data=importance_df, color='lightblue')
    
    # Add value labels
    for i, row in enumerate(importance_df.itertuples()):
        value_text = f"{row.Importance:.3f}"
        ax.text(row.Importance + 0.005, i, value_text, fontsize=10, va='center')
    
    plt.title('Random Forest Feature Importance', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    return best_model, scaler, best_params, best_accuracy, plt.gcf()


def perform_rf_feature_sensitivity_analysis(combined_data):
    """
    Perform sensitivity analysis by iteratively removing features
    based on Random Forest feature importance to find the best performing model.
    """
    print("\nStarting Random Forest Feature Sensitivity Analysis...")
    
    # Get features and target
    X = combined_data.drop(['type', 'quality'], axis=1)
    y = combined_data['type']
    
    # Get all feature names
    all_features = X.columns.tolist()
    
    # Initialize tracking variables
    best_accuracy = 0
    best_params = None
    best_features = all_features.copy()
    current_features = all_features.copy()
    
    # Initialize results tracking
    results = []
    
    # First, evaluate the model with all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    feature_order = X.columns.tolist()  # Store original feature order
    
    # Set up k-fold cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Define initial parameter grid - simplified for iteration
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Create base model
    rf = RandomForestClassifier(random_state=42)
    
    # Set up GridSearchCV for initial model
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=kf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    # Perform grid search
    grid_search.fit(X_scaled, y)
    
    # Record initial model with all features
    initial_best_params = grid_search.best_params_
    initial_accuracy = grid_search.best_score_
    initial_model = grid_search.best_estimator_
    
    best_accuracy = initial_accuracy
    best_params = initial_best_params.copy()
    
    results.append({
        'Features Removed': 'None',
        'Features Used': len(current_features),
        'Best Parameters': str(initial_best_params),
        'Accuracy': initial_accuracy
    })
    
    print(f"Initial model (all {len(current_features)} features): Accuracy = {initial_accuracy:.4f}")
    print(f"Parameters: {initial_best_params}")
    
    # Get feature importance from initial model
    feature_importance = dict(zip(all_features, initial_model.feature_importances_))
    
    # Create a list of features ordered by importance (least important first)
    features_by_importance = sorted(feature_importance.items(), key=lambda x: x[1])
    features_by_importance = [f[0] for f in features_by_importance]  # Extract just the feature names
    
    # Plot the initial feature importance
    initial_importance_df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    })
    initial_importance_df['Significant'] = initial_importance_df['Importance'] > np.mean(initial_importance_df['Importance'])
    initial_importance_df = initial_importance_df.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Importance', y='Feature', data=initial_importance_df, color='lightblue')
    
    # Add value labels
    for i, row in enumerate(initial_importance_df.itertuples()):
        value_text = f"{row.Importance:.3f}"
        ax.text(row.Importance + 0.005, i, value_text, fontsize=10, va='center')
    
    plt.title('Initial Random Forest Feature Importance', fontsize=16)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Wine Properties', fontsize=12)
    plt.tight_layout()
    
    initial_importance_fig = plt.gcf()
    
    print("\nInitial feature importance (lower = less important):")
    for feature in features_by_importance:
        print(f"  {feature}: {feature_importance[feature]:.6f}")
    
    # Storage for best model feature importance
    best_model_importance = {}
    best_model = initial_model
    
    # New code: Track the best scaler properly
    best_scaler = scaler  # Initialize with initial scaler
    
    # Now start removing features one by one, from least important to most important
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
        feature_order = current_features  # Update feature order for current iteration
        
        # Perform grid search with the reduced feature set
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=kf,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_scaled, y)
        
        current_best_params = grid_search.best_params_
        current_accuracy = grid_search.best_score_
        current_model = grid_search.best_estimator_
        
        # Record results
        results.append({
            'Features Removed': feature_to_remove,
            'Features Used': len(current_features),
            'Best Parameters': str(current_best_params),
            'Accuracy': current_accuracy
        })
        
        print(f"Removed '{feature_to_remove}': Accuracy = {current_accuracy:.4f}")
        
        # Update best model if this one is better
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_params = current_best_params.copy()
            best_features = current_features.copy()
            best_model = current_model
            best_scaler = scaler  # Save the scaler associated with the best model
            # Get feature importance for the best model
            best_model_importance = dict(zip(
                current_features, 
                current_model.feature_importances_
            ))
            print(f"  --> New best model found!")
            
        # Get new feature importance for next iteration
        if len(current_features) > 1:
            feature_importance = dict(zip(
                current_features, 
                current_model.feature_importances_
            ))
            features_by_importance = sorted(feature_importance.items(), key=lambda x: x[1])
            features_by_importance = [f[0] for f in features_by_importance]
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot the best model feature importance
    if best_model_importance:
        best_importance_df = pd.DataFrame({
            'Feature': list(best_model_importance.keys()),
            'Importance': list(best_model_importance.values())
        })
        best_importance_df['Significant'] = best_importance_df['Importance'] > np.mean(best_importance_df['Importance'])
        best_importance_df = best_importance_df.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='Importance', y='Feature', data=best_importance_df, color='lightblue')
        
        # Add value labels
        for i, row in enumerate(best_importance_df.itertuples()):
            value_text = f"{row.Importance:.3f}"
            ax.text(row.Importance + 0.005, i, value_text, fontsize=10, va='center')
        
        plt.title(f'Feature Importance for Best RF Model ({len(best_features)} features)', fontsize=16)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Wine Properties', fontsize=12)
        plt.tight_layout()
        
        best_importance_fig = plt.gcf()
    
    print("\nBest Model Summary:")
    print(f"Features: {best_features}")
    print(f"Number of features: {len(best_features)}")
    print(f"Best parameters: {best_params}")
    print(f"Cross-validated accuracy: {best_accuracy:.4f}")
    
    # Create visualization of accuracy vs features removed
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(results)), results_df['Accuracy'], 'o-', linewidth=2)
    plt.xlabel('Iteration (Features Removed)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Random Forest Model Accuracy vs. Features Removed', fontsize=16)
    plt.grid(True)
    
    # Add annotations for removed features
    for i, row in enumerate(results_df.itertuples()):
        if i > 0:  # Skip the first iteration (no features removed)
            plt.annotate(row._2,
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
    
    return best_model, best_scaler, best_features, best_params, best_accuracy, [initial_importance_fig, best_importance_fig, plt.gcf()]


def perform_anova_feature_sensitivity_analysis(combined_data, anova_results):
    """
    Perform sensitivity analysis by iteratively removing features
    based on ANOVA results to find the best performing Random Forest model.
    """
    print("\nStarting ANOVA-based Feature Sensitivity Analysis...")
    
    # Get features and target
    X = combined_data.drop(['type', 'quality'], axis=1)
    y = combined_data['type']
    
    # Get all feature names
    all_features = X.columns.tolist()
    
    # Create a list of features ordered by their importance (least important first)
    features_by_importance = anova_results['Variable'].tolist()
    features_by_importance.reverse()  # Reverse to get least important first
    
    # Initialize tracking variables
    best_accuracy = 0
    best_params = None
    best_features = all_features.copy()
    current_features = all_features.copy()
    
    # Initialize results tracking
    results = []
    
    # First, evaluate the model with all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    feature_order = X.columns.tolist()  # Store original feature order
    
    # Set up k-fold cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Define simplified parameter grid for iteration
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Create base model
    rf = RandomForestClassifier(random_state=42)
    
    # Set up GridSearchCV for initial model
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=kf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    # Perform grid search
    grid_search.fit(X_scaled, y)
    
    # Record initial model with all features
    initial_best_params = grid_search.best_params_
    initial_accuracy = grid_search.best_score_
    initial_model = grid_search.best_estimator_
    
    best_accuracy = initial_accuracy
    best_params = initial_best_params.copy()
    best_model = initial_model
    
    results.append({
        'Features Removed': 'None',
        'Features Used': len(current_features),
        'Best Parameters': str(initial_best_params),
        'Accuracy': initial_accuracy
    })
    
    print(f"Initial model (all {len(current_features)} features): Accuracy = {initial_accuracy:.4f}")
    print(f"Parameters: {initial_best_params}")
    
    # Initialize best scaler
    best_scaler = scaler
    
    # Now start removing features one by one based on ANOVA importance
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
        feature_order = current_features  # Update feature order for current iteration
        
        # Perform grid search with the reduced feature set
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=kf,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_scaled, y)
        
        current_best_params = grid_search.best_params_
        current_accuracy = grid_search.best_score_
        current_model = grid_search.best_estimator_
        
        # Record results
        results.append({
            'Features Removed': feature_to_remove,
            'Features Used': len(current_features),
            'Best Parameters': str(current_best_params),
            'Accuracy': current_accuracy
        })
        
        print(f"Removed '{feature_to_remove}': Accuracy = {current_accuracy:.4f}")
        
        # Update best model if this one is better
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_params = current_best_params.copy()
            best_features = current_features.copy()
            best_model = current_model
            best_scaler = scaler  # Save the scaler associated with the best model
            print(f"  --> New best model found!")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    print("\nBest Model Summary:")
    print(f"Features: {best_features}")
    print(f"Number of features: {len(best_features)}")
    print(f"Best parameters: {best_params}")
    print(f"Cross-validated accuracy: {best_accuracy:.4f}")
    
    # Create visualization of accuracy vs features removed
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(results)), results_df['Accuracy'], 'o-', linewidth=2)
    plt.xlabel('Iteration (Features Removed)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Random Forest Model Accuracy vs. Features Removed (Using ANOVA)', fontsize=16)
    plt.grid(True)
    
    # Add annotations for removed features
    for i, row in enumerate(results_df.itertuples()):
        if i > 0:  # Skip the first iteration (no features removed)
            plt.annotate(row._2,
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
    return best_model, best_scaler, best_features, best_params, best_accuracy, plt.gcf()


def create_confusion_matrix(model, X, y, scaler, title):
    """
    Create and display a confusion matrix for the given model.
    
    Parameters:
    -----------
    model : sklearn model
        The trained model to evaluate
    X : DataFrame
        Features for prediction
    y : Series
        True labels
    scaler : StandardScaler
        Scaler used to preprocess the data
    title : str
        Title for the confusion matrix plot
    """
    # Preprocess the data
    X_scaled = scaler.transform(X)
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    
    # Create confusion matrix
    cm = confusion_matrix(y, y_pred, labels=['white', 'red'])
    
    # Create plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['white', 'red'], 
                yticklabels=['white', 'red'])
    
    plt.title(f"{title}\nAccuracy: {accuracy:.4f}", fontsize=14)
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.tight_layout()
    
    return plt.gcf()


def plot_decision_boundary(combined_data, model, scaler, features, accuracy, title="Random Forest Model Decision Boundary"):
    """
    Plot the decision boundary for the Random Forest model using the two most important features.
    
    Parameters:
    -----------
    combined_data : pandas DataFrame
        The dataset containing all wine features
    model : RandomForestClassifier
        The trained Random Forest model
    scaler : StandardScaler
        The scaler used to standardize features
    features : list
        List of features used in the model
    accuracy : float
        The accuracy of the model
    title : str
        Title for the plot
    """
    # If we have more than 2 features, select the 2 most important ones for visualization
    if len(features) > 2:
        # Get feature importance
        importances = model.feature_importances_
        # Create a dictionary of feature importances
        feature_importance = dict(zip(features, importances))
        # Sort by importance (descending)
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        # Get the two most important features
        top_features = [sorted_features[0][0], sorted_features[1][0]]
    else:
        top_features = features
    
    print(f"Plotting decision boundary using features: {top_features}")
    
    # Get the features and target
    X = combined_data[top_features]
    y = combined_data['type']
    
    # Create a meshgrid for plotting the decision boundary
    x_min, x_max = X[top_features[0]].min() - 0.1, X[top_features[0]].max() + 0.1
    y_min, y_max = X[top_features[1]].min() - 0.1, X[top_features[1]].max() + 0.1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min) / 100),
                         np.arange(y_min, y_max, (y_max - y_min) / 100))
    
    # For prediction, we need data in the format of all features
    if len(features) > 2:
        # Create a dataframe with the mean values for all features
        grid = pd.DataFrame(columns=features)
        
        # Create a meshgrid for our 2D grid
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        total_points = mesh_points.shape[0]
        
        # Initialize with mean values
        for feature in features:
            grid[feature] = np.repeat(combined_data[feature].mean(), total_points)
        
        # Set the values for our two visualization features
        grid[top_features[0]] = mesh_points[:, 0]
        grid[top_features[1]] = mesh_points[:, 1]
        
        # Scale the grid points
        grid_scaled = scaler.transform(grid)
        
        # Predict classes
        Z = model.predict(grid_scaled)
    else:
        # If we're just using 2 features, it's simpler
        grid = pd.DataFrame(columns=features)
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        grid[features[0]] = mesh_points[:, 0]
        grid[features[1]] = mesh_points[:, 1]
        
        grid_scaled = scaler.transform(grid)
        Z = model.predict(grid_scaled)
    
    # Convert to numeric values (0 for white, 1 for red)
    Z_numeric = np.array([1 if z == 'red' else 0 for z in Z])
    Z_numeric = Z_numeric.reshape(xx.shape)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z_numeric, alpha=0.3, cmap=plt.cm.RdBu_r)
    
    # Plot the data points
    scatter = plt.scatter(X[top_features[0]], X[top_features[1]], 
                c=[0 if t == 'white' else 1 for t in y], 
                cmap=plt.cm.RdBu_r, edgecolors='k', s=40, alpha=0.7)
    
    plt.xlabel(top_features[0], fontsize=12)
    plt.ylabel(top_features[1], fontsize=12)
    plt.title(f'{title} (accuracy={accuracy:.4f})', fontsize=16)
    
    # Add color legend
    labels = ['White Wine', 'Red Wine']
    plt.legend(handles=scatter.legend_elements()[0], labels=labels)
    
    plt.tight_layout()
    return plt.gcf()


def plot_simple_rf_decision_boundary(combined_data, model, scaler, accuracy, top_features=None):
    """
    Plot a simplified decision boundary for the RF model using the two most important features
    or user-specified features.
    
    Parameters:
    -----------
    combined_data : pandas DataFrame
        The dataset containing all wine features
    model : RandomForestClassifier
        The trained RF model
    scaler : StandardScaler
        The scaler used to standardize features
    accuracy : float
        The accuracy of the model
    top_features : list of str, optional
        Two features to use for plotting; if None, uses the two most important features
    """
    # If specific features aren't provided, use the top 2 features from feature importance
    if top_features is None:
        # Get feature importance from model
        X_full = combined_data.drop(['type', 'quality'], axis=1)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = X_full.columns
        top_features = [features[indices[0]], features[indices[1]]]
    
    print(f"Plotting simple decision boundary using features: {top_features}")
    
    # Get the data with only the two specified features
    X = combined_data[top_features]
    y = combined_data['type']
    
    # Create a meshgrid for plotting the decision boundary
    x_min, x_max = X[top_features[0]].min() - 0.1, X[top_features[0]].max() + 0.1
    y_min, y_max = X[top_features[1]].min() - 0.1, X[top_features[1]].max() + 0.1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min) / 100),
                         np.arange(y_min, y_max, (y_max - y_min) / 100))
    
    # Create a dataframe for prediction
    grid = pd.DataFrame({
        top_features[0]: xx.ravel(),
        top_features[1]: yy.ravel()
    })
    
    # Scale the grid points
    grid_scaled = scaler.transform(grid)
    
    # Predict classes
    Z = model.predict(grid_scaled)
    
    # Convert to numeric values (0 for white, 1 for red)
    Z_numeric = np.array([1 if z == 'red' else 0 for z in Z])
    Z_numeric = Z_numeric.reshape(xx.shape)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z_numeric, alpha=0.3, cmap=plt.cm.RdBu_r)
    
    # Plot the data points
    scatter = plt.scatter(X[top_features[0]], X[top_features[1]], 
                c=[0 if t == 'white' else 1 for t in y], 
                cmap=plt.cm.RdBu_r, edgecolors='k', s=40, alpha=0.7)
    
    plt.xlabel(top_features[0], fontsize=12)
    plt.ylabel(top_features[1], fontsize=12)
    plt.title(f'Simple Random Forest Decision Boundary (2 features, accuracy={accuracy:.4f})', fontsize=16)
    
    # Add color legend
    labels = ['White Wine', 'Red Wine']
    plt.legend(handles=scatter.legend_elements()[0], labels=labels)
    
    plt.tight_layout()
    return plt.gcf()

# Load the datasets - modify this section
# Get the directory where your script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
white_wine_path = os.path.join(script_dir, 'winequality-white.csv')
red_wine_path = os.path.join(script_dir, 'winequality-red.csv')

white_wine_data = pd.read_csv(white_wine_path, sep=';')
red_wine_data = pd.read_csv(red_wine_path, sep=';')

# Add column to indicate wine type
white_wine_data['type'] = 'white'
red_wine_data['type'] = 'red'

# Combine datasets
combined_data = pd.concat([white_wine_data, red_wine_data], axis=0)

# Perform ANOVA analysis
print('Starting ANOVA analysis...')
anova_results, anova_fig = analyze_anova(combined_data)

# Train the full Random Forest model
print('Training full Random Forest model...')
rf_model, scaler, best_params, best_accuracy, importance_fig = train_rf_model(combined_data)

# Create confusion matrix for the full model
print('Creating confusion matrix for the full model...')
X_full = combined_data.drop(['type', 'quality'], axis=1)
y_full = combined_data['type']
full_model_cm = create_confusion_matrix(rf_model, X_full, y_full, scaler, 
                                       "Full Random Forest Model Confusion Matrix")

# Perform RF-based feature sensitivity analysis
print('Starting RF-based sensitivity analysis...')
rf_best_model, rf_best_scaler, rf_best_features, rf_best_params, rf_best_accuracy, rf_figures = perform_rf_feature_sensitivity_analysis(combined_data)

# Unpack the RF feature importance figures
rf_initial_importance_fig, rf_best_importance_fig, rf_sensitivity_fig = rf_figures

# Create confusion matrix for the RF-based feature sensitivity model
print('Creating confusion matrix for the RF-based feature sensitivity model...')
X_rf_best = combined_data[rf_best_features]
rf_best_model_cm = create_confusion_matrix(rf_best_model, X_rf_best, y_full, rf_best_scaler,
                                         f"Feature Importance-Based Model Confusion Matrix\n({len(rf_best_features)} features)")

# Perform ANOVA-based feature sensitivity analysis
print('Starting ANOVA-based feature sensitivity analysis...')
best_model, best_scaler, best_features, best_params, best_accuracy, sensitivity_fig = perform_anova_feature_sensitivity_analysis(combined_data, anova_results)

# Create confusion matrix for the ANOVA-based feature sensitivity model
print('Creating confusion matrix for the ANOVA-based feature sensitivity model...')
X_anova_best = combined_data[best_features]
anova_best_model_cm = create_confusion_matrix(best_model, X_anova_best, y_full, best_scaler,
                                            f"ANOVA-Based Model Confusion Matrix\n({len(best_features)} features)")



# For the full model
print('Creating decision boundary for the full RF model...')
X_full = combined_data.drop(['type', 'quality'], axis=1)
y_full = combined_data['type']
full_model_decision_boundary = plot_decision_boundary(
    combined_data, rf_model, scaler, X_full.columns.tolist(), best_accuracy, 
    title="Full Random Forest Model Decision Boundary"
)

# For the RF-based feature sensitivity model
print('Creating decision boundary for the RF-based feature sensitivity model...')
rf_sensitivity_decision_boundary = plot_decision_boundary(
    combined_data, rf_best_model, rf_best_scaler, rf_best_features, rf_best_accuracy,
    title="Feature Importance-Based RF Model Decision Boundary"
)

# For the ANOVA-based feature sensitivity model
print('Creating decision boundary for the ANOVA-based feature sensitivity model...')
anova_sensitivity_decision_boundary = plot_decision_boundary(
    combined_data, best_model, best_scaler, best_features, best_accuracy,
    title="ANOVA-Based RF Model Decision Boundary"
)


plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')