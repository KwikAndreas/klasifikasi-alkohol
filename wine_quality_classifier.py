import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the wine quality dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    print(f"Dataset shape: {data.shape}")
    print(data.head())
    return data

# Exploratory Data Analysis
def explore_data(data):
    print("\nBasic Statistics:")
    print(data.describe())
    
    print("\nMissing values:")
    print(data.isnull().sum())
    
    print("\nClass distribution:")
    print(data['Class'].value_counts())
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=data, x='Sugar', y='pH', hue='Class')
    plt.title('Sugar vs pH by Wine Quality')
    
    plt.subplot(1, 2, 2)
    sns.countplot(data=data, x='Class')
    plt.title('Distribution of Wine Quality Classes')
    
    plt.tight_layout()
    plt.savefig('wine_quality_eda.png')
    plt.close()

# Feature Engineering
def engineer_features(data):
    """Create new features that might help improve model performance"""
    # Create a copy of the data
    enhanced_data = data.copy()
    
    # Add polynomial features
    enhanced_data['Sugar_Squared'] = enhanced_data['Sugar'] ** 2
    enhanced_data['pH_Squared'] = enhanced_data['pH'] ** 2
    enhanced_data['Sugar_pH_Interaction'] = enhanced_data['Sugar'] * enhanced_data['pH']
    
    # Add logarithmic features (adding small value to avoid log(0))
    enhanced_data['Log_Sugar'] = np.log1p(enhanced_data['Sugar'])
    
    # Add ratio feature
    enhanced_data['Sugar_to_pH_Ratio'] = enhanced_data['Sugar'] / enhanced_data['pH']
    
    # Binning features
    enhanced_data['Sugar_Bin'] = pd.qcut(enhanced_data['Sugar'], 4, labels=False)
    enhanced_data['pH_Bin'] = pd.qcut(enhanced_data['pH'], 4, labels=False)
    
    # Convert categorical bins to one-hot encoding
    sugar_bins = pd.get_dummies(enhanced_data['Sugar_Bin'], prefix='Sugar_Bin')
    ph_bins = pd.get_dummies(enhanced_data['pH_Bin'], prefix='pH_Bin')
    
    # Combine with original dataframe
    enhanced_data = pd.concat([enhanced_data, sugar_bins, ph_bins], axis=1)
    
    # Drop the intermediate bin columns
    enhanced_data.drop(['Sugar_Bin', 'pH_Bin'], axis=1, inplace=True)
    
    print(f"Original features: {list(data.columns)}")
    print(f"Enhanced features: {list(enhanced_data.columns)}")
    
    return enhanced_data

# Preprocess the data
def preprocess_data(data, use_feature_engineering=True):
    """Preprocess the data for model training"""
    # Apply feature engineering if enabled
    if use_feature_engineering:
        data = engineer_features(data)
    
    # Convert class labels to numeric values
    data['Class'] = data['Class'].map({'Good': 1, 'Poor': 0})
    
    # Define features and target
    y = data['Class']
    X = data.drop('Class', axis=1)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Store column names for feature importance
    feature_names = X.columns.tolist()
    
    return X_train, X_test, y_train, y_test, scaler, feature_names

# Train multiple models and select the best one
def train_models(X_train, y_train, feature_names):
    """Train multiple models with hyperparameter tuning"""
    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define base models
    base_models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'KNN': KNeighborsClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    # Define parameter grids for tuning
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': [None, 'balanced']
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'linear', 'poly'],
            'class_weight': [None, 'balanced']
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 11, 13],
            'weights': ['uniform', 'distance'],
            'p': [1, 2],
            'leaf_size': [20, 30, 40]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 4],
            'min_samples_leaf': [1, 2],
            'subsample': [0.8, 0.9, 1.0]
        },
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'solver': ['saga', 'liblinear'],
            'class_weight': [None, 'balanced']
        },
        'AdaBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.5, 1.0],
            'algorithm': ['SAMME', 'SAMME.R']
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2],
            'scale_pos_weight': [1, 3, 5]  # For imbalanced classes
        }
    }
    
    best_models = {}
    
    # Train individual models
    for name, model in base_models.items():
        print(f"\nTraining {name}...")
        grid = GridSearchCV(
            model, 
            param_grids[name], 
            cv=cv, 
            scoring='accuracy', 
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train, y_train)
        
        best_models[name] = grid.best_estimator_
        print(f"Best parameters for {name}: {grid.best_params_}")
        print(f"Best cross-validation accuracy: {grid.best_score_:.4f}")
        
        # For models with feature_importances_ attribute, show feature importance
        if hasattr(grid.best_estimator_, 'feature_importances_'):
            importances = grid.best_estimator_.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print(f"\nFeature ranking for {name}:")
            for i, idx in enumerate(indices):
                if i < 10:  # Show top 10 features
                    print(f"{i+1}. {feature_names[idx]} ({importances[idx]:.4f})")
    
    # Create a voting classifier from the best models
    # Include XGBoost in the voting ensemble
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', best_models['Random Forest']),
            ('svm', best_models['SVM']),
            ('gb', best_models['Gradient Boosting']),
            ('xgb', best_models['XGBoost']),
            ('lr', best_models['Logistic Regression'])
        ],
        voting='soft'
    )
    
    # Train the voting classifier
    print("\nTraining Voting Classifier...")
    voting_clf.fit(X_train, y_train)
    best_models['Voting Classifier'] = voting_clf
    
    return best_models

# Evaluate the models
def evaluate_models(models, X_test, y_test, feature_names):
    """Evaluate models and return results"""
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate ROC AUC for probabilistic models
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        except:
            roc_auc = None
        
        results[name] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'model': model
        }
        
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        if roc_auc:
            print(f"ROC AUC: {roc_auc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{name.replace(" ", "_")}.png')
        plt.close()
    
    return results

# Plot the decision boundaries
def plot_decision_boundaries(model, X_test, y_test, model_name):
    """Plot decision boundaries for the model"""
    # Create a mesh grid
    h = 0.02  # step size in the mesh
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # For 2D visualization, we'll use only the first 2 features
    subset_data = np.c_[xx.ravel(), yy.ravel()]
    pad_width = max(0, X_test.shape[1] - 2)
    if pad_width > 0:
        # Pad with zeros if we have more than 2 features
        pad_data = np.zeros((subset_data.shape[0], pad_width))
        subset_data = np.hstack((subset_data, pad_data))
    
    # Predict the class for each point in mesh
    Z = model.predict(subset_data)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    
    # Plot the first two dimensions of test points
    scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', cmap=plt.cm.RdYlBu)
    plt.xlabel('Feature 1 (Scaled)')
    plt.ylabel('Feature 2 (Scaled)')
    plt.title(f'Decision Boundary - {model_name}')
    plt.legend(*scatter.legend_elements(), title='Classes')
    
    plt.savefig(f'decision_boundary_{model_name.replace(" ", "_")}.png')
    plt.close()

# Find the best model
def get_best_model(results):
    """Return the best model based on accuracy"""
    best_accuracy = 0
    best_model_name = None
    
    for name, result in results.items():
        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            best_model_name = name
    
    return best_model_name, results[best_model_name]

# Main function to run the classification pipeline
def main():
    file_path = "d:/Latihan/machine_learning/alkohol/data/wine_quality.csv"
    
    # Load and explore the data
    data = load_data(file_path)
    explore_data(data)
    
    # Try both with and without feature engineering to compare results
    approaches = [
        {"name": "Basic Features", "use_feature_engineering": False},
        {"name": "Enhanced Features", "use_feature_engineering": True}
    ]
    
    best_overall_accuracy = 0
    best_overall_model = None
    best_overall_scaler = None
    best_feature_names = None
    best_approach_name = None
    
    for approach in approaches:
        print(f"\n\n{'='*80}")
        print(f"Trying approach: {approach['name']}")
        print(f"{'='*80}")
        
        # Preprocess the data
        X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(
            data, use_feature_engineering=approach["use_feature_engineering"]
        )
        
        # Train multiple models
        models = train_models(X_train, y_train, feature_names)
        
        # Evaluate the models
        results = evaluate_models(models, X_test, y_test, feature_names)
        
        # Get the best model for this approach
        best_model_name, best_model_result = get_best_model(results)
        
        print(f"\nBest Model for {approach['name']}: {best_model_name}")
        print(f"Best Model Accuracy: {best_model_result['accuracy']:.4f}")
        
        # Plot decision boundary for the best model in this approach
        if X_test.shape[1] >= 2:  # Can only plot if we have at least 2 features
            plot_decision_boundaries(
                best_model_result['model'], 
                X_test[:, :2],  # Use first 2 features for visualization
                y_test, 
                f"{approach['name']}_{best_model_name}"
            )
        
        # Track the overall best model
        if best_model_result['accuracy'] > best_overall_accuracy:
            best_overall_accuracy = best_model_result['accuracy']
            best_overall_model = best_model_result['model']
            best_overall_scaler = scaler
            best_feature_names = feature_names
            best_approach_name = f"{approach['name']}_{best_model_name}"
    
    # Final report on the best overall model
    print(f"\n\n{'='*80}")
    print(f"Final Results")
    print(f"{'='*80}")
    print(f"Best Overall Approach: {best_approach_name}")
    print(f"Best Overall Accuracy: {best_overall_accuracy:.4f}")
    
    # Check if accuracy meets the threshold
    if best_overall_accuracy >= 0.75:
        print(f"\nSuccess! The model has achieved an accuracy of {best_overall_accuracy:.4f}, which is above 75%!")
    else:
        print(f"\nThe model accuracy is {best_overall_accuracy:.4f}, which is below 75%. Consider adding more data or trying other techniques.")
    
    # Save the best model
    joblib.dump(best_overall_model, 'wine_quality_classifier_model.joblib')
    joblib.dump(scaler, 'wine_quality_scaler.joblib')
    
    # Save feature names for later use in prediction
    with open('feature_names.txt', 'w') as f:
        f.write('\n'.join(best_feature_names))
    
    print(f"\nModel saved successfully as 'wine_quality_classifier_model.joblib'.")
    print(f"Scaler saved as 'wine_quality_scaler.joblib'.")
    print(f"Feature names saved as 'feature_names.txt'.")
    print("\nYou can use these to predict wine quality using Sugar and pH values.")

if __name__ == "__main__":
    main()
