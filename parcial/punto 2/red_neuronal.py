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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from scipy.stats import mode, ttest_ind, chi2_contingency, ks_2samp
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from matplotlib.colors import ListedColormap
import itertools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier

# Define a subclass to explicitly set the estimator type
class MyKerasClassifier(KerasClassifier):
    _estimator_type = "classifier"

def full_model(shape):
    model = Sequential([
        Dense(8, activation='relu', input_shape=(shape,)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.01)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_complete_model(df):
    # Define features (X) and target (y)
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']
    input_shape = X.shape[1]

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # Create KerasClassifier wrapper using the subclass
    keras_classifier = MyKerasClassifier(
        model=lambda: full_model(input_shape), # Use lambda to pass the function correctly
        loss="binary_crossentropy",
        optimizer="adam",           # Optimizer can be specified here or in full_model compile
        optimizer__learning_rate=0.01, # Access optimizer parameters like this
        metrics=["accuracy"],
        epochs=50,
        batch_size=128,
        validation_split=0.2,
        callbacks=[early_stopping],
        random_state=42,
        verbose=1
    )

    # Train the KerasClassifier
    print("Training KerasClassifier...")
    # Scikit-learn fit expects numpy arrays ideally
    X_np = X.values if isinstance(X, pd.DataFrame) else X
    y_np = y.values if isinstance(y, pd.Series) else y
    keras_classifier.fit(X_np, y_np)

    # The History object is stored in keras_classifier.history_
    history = keras_classifier.history_

    # Evaluate on the training data using the wrapper's score method
    # Note: score gives accuracy by default for classifiers
    accuracy = keras_classifier.score(X_np, y_np)
    # Loss needs to be calculated differently if needed, e.g., using evaluate method on the underlying Keras model if accessible or manually
    # For simplicity, let's retrieve the final training loss from history
    loss = history['loss'][-1] # Approximate loss, or re-evaluate if exact needed

    # Get predictions using the wrapper
    y_pred = keras_classifier.predict(X_np)
    y_pred_proba = keras_classifier.predict_proba(X_np)
    y_pred_proba_positive = y_pred_proba[:, 1] # Probability of the positive class (usually index 1)


    # Calculate metrics based on predictions
    precision = precision_score(y_np, y_pred, zero_division=0)
    recall = recall_score(y_np, y_pred)
    f1 = f1_score(y_np, y_pred)
    auc = roc_auc_score(y_np, y_pred_proba_positive) # Use positive class probability for AUC

    # Calculate KS Statistic
    proba_positive = y_pred_proba_positive[y_np == 1]
    proba_negative = y_pred_proba_positive[y_np == 0]
    ks_statistic, _ = ks_2samp(proba_positive, proba_negative)

    # Store results
    results = {
        'keras_classifier': keras_classifier, # Store the fitted wrapper
        'history_dict': history,             # Store the history dictionary
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'ks': ks_statistic,
        'loss': loss,
        'X': X,
        'y': y,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba_positive, # Store positive class probability
    }

    return results

def plot_training_results(results, filepath):
    history_dict = results['history_dict'] # Get history from results

    # Create separate figure for loss
    plt.figure(figsize=(12, 8))
    plt.title('Pérdida durante el entrenamiento', fontsize=14)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Pérdida', fontsize=12)
    plt.grid(True)

    plt.plot(history_dict['loss'], color='blue', label='Train loss')
    plt.plot(history_dict['val_loss'], color='blue',
            alpha=0.5, linestyle='--', label='Val loss')

    plt.ylim(bottom=0) # Adjust ylim if needed, maybe start from 0
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(filepath, 'loss_completo.png'), dpi=300)


    # Create separate figure for accuracy
    plt.figure(figsize=(12, 8))
    plt.title('Accuracy durante el entrenamiento', fontsize=14) # Changed title
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12) # Changed y-axis label
    plt.grid(True)

    plt.plot(history_dict['accuracy'], color='green', label='Train accuracy')
    plt.plot(history_dict['val_accuracy'], color='green',
            alpha=0.5, linestyle='--', label='Val accuracy')

    plt.ylim(0, 1)  # Set y-axis from 0 to 1
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(filepath, 'accuracy_completo.png'), dpi=300)

def plot_confusion_matrix(results, filepath ):
    y = results['y']
    y_pred = results['y_pred']

    # Calculate confusion matrix
    cm = confusion_matrix(y, y_pred)

    # Create figure and plot
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                 display_labels=['No Diabetes', 'Diabetes']) # Adjusted labels
    disp.plot(cmap=plt.cm.Blues, values_format='d')

    plt.title('Matriz de Confusión - Modelo Completo')
    plt.tight_layout()
    plt.savefig(os.path.join(filepath, 'confusion_matrix_completo.png'), dpi=300)

def plot_model_accuracy(results, filepath):
    y_true = results['y']
    y_pred = results['y_pred']
    y_pred_proba = results['y_pred_proba']

    # Calculate metrics (ensure they are calculated based on y_true and y_pred/y_pred_proba)
    accuracy = results.get('accuracy', accuracy_score(y_true, y_pred))
    f1 = results.get('f1', f1_score(y_true, y_pred))
    precision = results.get('precision', precision_score(y_true, y_pred, zero_division=0))
    recall = results.get('recall', recall_score(y_true, y_pred))
    auc = results.get('auc', roc_auc_score(y_true, y_pred_proba))

    # Get or calculate KS
    if 'ks' in results:
        ks = results['ks']
    else:
        # Need y_true to be boolean or 0/1 for this indexing
        y_true_bool = y_true.astype(bool) if isinstance(y_true, pd.Series) else y_true.astype(bool)
        # Check if y_pred_proba is 1D or 2D
        if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] >= 1:
             # Assuming positive class probability is already selected
             proba_positive = y_pred_proba[y_true_bool] # Get probabilities where true label is 1
             proba_negative = y_pred_proba[~y_true_bool] # Get probabilities where true label is 0
        else: # Handle case if y_pred_proba is somehow 1D
             proba_positive = y_pred_proba[y_true_bool]
             proba_negative = y_pred_proba[~y_true_bool]
        ks, _ = ks_2samp(proba_positive, proba_negative)


    # Create figure
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')  # Hide axes
    ax.axis('tight')  # Tight layout

    # Create data for table
    metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'AUC', 'KS Stat']
    values = [f'{accuracy:.4f}', f'{f1:.4f}', f'{precision:.4f}', f'{recall:.4f}', f'{auc:.4f}', f'{ks:.4f}']

    # Create table with colored cells based on value
    cell_colors = []
    # Accuracy, F1, Precision, Recall coloring
    for val in [accuracy, f1, precision, recall]:
        if val < 0.7: color = (1.0, 0.7, 0.7)  # light red
        elif val < 0.8: color = (1.0, 0.9, 0.7)  # yellow
        elif val < 0.9: color = (0.8, 1.0, 0.8)  # light green
        else: color = (0.6, 1.0, 0.6)  # green
        cell_colors.append(color)
    # AUC coloring
    if auc < 0.7: color = (1.0, 0.7, 0.7)
    elif auc < 0.8: color = (1.0, 0.9, 0.7)
    elif auc < 0.9: color = (0.8, 1.0, 0.8)
    else: color = (0.6, 1.0, 0.6)
    cell_colors.append(color)
    # KS coloring
    if ks < 0.3: color = (1.0, 0.7, 0.7)  # Poor
    elif ks < 0.5: color = (1.0, 0.9, 0.7)  # Fair/Good
    elif ks < 0.7: color = (0.8, 1.0, 0.8)  # Good/Very Good
    else: color = (0.6, 1.0, 0.6) # Excellent
    cell_colors.append(color)

    table = ax.table(
        cellText=[values],
        colLabels=metrics,
        loc='center',
        cellLoc='center',
        cellColours=[cell_colors]
    )

    # Customize table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.1, 1.8)

    plt.title('Métricas del Modelo', fontsize=14, pad=20)
    plt.tight_layout()

    # Save figure with the specified filename
    plt.savefig(os.path.join(filepath, 'model_metrics.png'), dpi=300)



# Set up directories
current_dir = os.path.dirname(os.path.abspath(__file__))
graficos_dir_base = os.path.join(current_dir, 'graficos', 'red_neuronal')
os.makedirs(graficos_dir_base, exist_ok=True) # Ensure the base directory exists

# Define subdirectories within the base directory
graficos_dir_completo = os.path.join(graficos_dir_base, 'completo')
graficos_dir_modelo2 = os.path.join(graficos_dir_base, 'modelo2')
graficos_dir_modelo3 = os.path.join(graficos_dir_base, 'modelo3')
graficos_dir_modelo4 = os.path.join(graficos_dir_base, 'modelo4')

# Create the subdirectories
os.makedirs(graficos_dir_completo, exist_ok=True)
os.makedirs(graficos_dir_modelo2, exist_ok=True)
os.makedirs(graficos_dir_modelo3, exist_ok=True)
os.makedirs(graficos_dir_modelo4, exist_ok=True)


df = pd.read_csv('punto 2/diabetes_binary_health_indicators_BRFSS2015_transformed.csv')


print(df.info())

# modelo completo
results = train_complete_model(df)
# Print results (Precision, Recall, F1 are now in results)
print(f"Model Accuracy: {results['accuracy']:.4f}")
print(f"Model Precision: {results['precision']:.4f}")
print(f"Model Recall: {results['recall']:.4f}")
print(f"Model F1 Score: {results['f1']:.4f}")
print("\nGenerating plots for the full model...")
plot_training_results(results,graficos_dir_completo)
plot_confusion_matrix(results,graficos_dir_completo)
plot_model_accuracy(results,graficos_dir_completo)


#modelo 2
print("-- Arrancando modelo 2 --")
features_model2 = [
    'HighBP', 'HighChol', 'DiffWalk',
    # BMI Categories
    'BMI_Category_Normal', 'BMI_Category_Overweight', 'BMI_Category_Obese_I',
    'BMI_Category_Obese_II', 'BMI_Category_Obese_III',
    # Age Categories
    'Age_2', 'Age_3', 'Age_4', 'Age_5', 'Age_6', 'Age_7', 'Age_8', 'Age_9',
    'Age_10', 'Age_11', 'Age_12', 'Age_13',
    # GenHlth Categories
    'GenHlth_2', 'GenHlth_3', 'GenHlth_4', 'GenHlth_5'
]
columns_for_df2 = ['Diabetes_binary'] + features_model2
df2 = df[columns_for_df2].copy()

results_model2 = train_complete_model(df2)
# Print results (Precision, Recall, F1 are now in results)
print(f"Model Accuracy: {results_model2['accuracy']:.4f}")
print(f"Model Precision: {results_model2['precision']:.4f}")
print(f"Model Recall: {results_model2['recall']:.4f}")
print(f"Model F1 Score: {results_model2['f1']:.4f}")
print("\nGenerating plots for the full model...")
plot_training_results(results_model2,graficos_dir_modelo2)
plot_confusion_matrix(results_model2,graficos_dir_modelo2)
plot_model_accuracy(results_model2,graficos_dir_modelo2)


#modelo 3
print("-- Arrancando modelo 3 --")
features_model3 = [
    'CholCheck', 'PhysActivity',
    # BMI Categories
    'BMI_Category_Normal', 'BMI_Category_Overweight', 'BMI_Category_Obese_I',
    'BMI_Category_Obese_II', 'BMI_Category_Obese_III',
    # Age Categories
    'Age_2', 'Age_3', 'Age_4', 'Age_5', 'Age_6', 'Age_7', 'Age_8', 'Age_9',
    'Age_10', 'Age_11', 'Age_12', 'Age_13',
    # Income
    'Income_2', 'Income_3', 'Income_4', 'Income_5', 'Income_6', 'Income_7', 'Income_8',
]
columns_for_df3 = ['Diabetes_binary'] + features_model3
df3 = df[columns_for_df3].copy()

results_model3 = train_complete_model(df3)
# Print results (Precision, Recall, F1 are now in results)
print(f"Model Accuracy: {results_model3['accuracy']:.4f}")
print(f"Model Precision: {results_model3['precision']:.4f}")
print(f"Model Recall: {results_model3['recall']:.4f}")
print(f"Model F1 Score: {results_model3['f1']:.4f}")
print("\nGenerating plots for the full model...")
plot_training_results(results_model3,graficos_dir_modelo3)
plot_confusion_matrix(results_model3,graficos_dir_modelo3)
plot_model_accuracy(results_model3,graficos_dir_modelo3)


#modelo 4
print("-- Arrancando modelo 4 --")
features_model4 = [
    'HighChol', 'DiffWalk', 'PhysActivity',
    # GenHlth Categories
    'GenHlth_2', 'GenHlth_3', 'GenHlth_4', 'GenHlth_5',
    # Age Categories
    'Age_2', 'Age_3', 'Age_4', 'Age_5', 'Age_6', 'Age_7', 'Age_8', 'Age_9',
    'Age_10', 'Age_11', 'Age_12', 'Age_13',
    # Income
    'Income_2', 'Income_3', 'Income_4', 'Income_5', 'Income_6', 'Income_7', 'Income_8'
]
columns_for_df4 = ['Diabetes_binary'] + features_model4
df4 = df[columns_for_df4].copy()

results_model4 = train_complete_model(df4)
# Print results (Precision, Recall, F1 are now in results)
print(f"Model Accuracy: {results_model4['accuracy']:.4f}")
print(f"Model Precision: {results_model4['precision']:.4f}")
print(f"Model Recall: {results_model4['recall']:.4f}")
print(f"Model F1 Score: {results_model4['f1']:.4f}")
print("\nGenerating plots for the full model...")
plot_training_results(results_model4,graficos_dir_modelo4)
plot_confusion_matrix(results_model4,graficos_dir_modelo4)
plot_model_accuracy(results_model4,graficos_dir_modelo4)



# Show plots
plt.show(block=False)
input("\nPresiona Enter para finalizar el programa y cerrar todas las figuras...")
print("Programa finalizado. Cerrando figuras...")
plt.close('all')
