# Wine Quality Classifier

This project implements a machine learning model to classify wine quality as "Good" or "Poor" based on Sugar and pH values.

## Project Structure

- `wine_quality_classifier.py`: Main script to train and evaluate the classification model
- `predict_wine_quality.py`: Script to make predictions using the trained model
- `wine_quality_classifier_model.joblib`: Saved model file (generated after training)
- `wine_quality_scaler.joblib`: Feature scaler (generated after training)
- `feature_names.txt`: List of feature names used by the model (generated after training)
- `data/wine_quality.csv`: Dataset containing wine features and quality labels
- `requirements.txt`: List of Python dependencies
- `setup_venv.bat`: Batch script to set up virtual environment (Windows)
- `setup_venv.sh`: Shell script to set up virtual environment (Linux/Mac)

## Setting Up the Environment

### Using Virtual Environment (Recommended)

#### Windows:

```bash
# Run the setup script
setup_venv.bat

# Activate the virtual environment
venv\Scripts\activate
```

#### Linux/Mac:

```bash
# Make the setup script executable
chmod +x setup_venv.sh

# Run the setup script
./setup_venv.sh

# Activate the virtual environment
source venv/bin/activate
```

### Manual Setup:

```bash
# Create virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

The following packages are required and will be installed in the virtual environment:

- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- joblib
- xgboost

## Usage

### Training the Model

To train the model, run:

```bash
python wine_quality_classifier.py
```

This will:

1. Load the wine quality dataset
2. Perform exploratory data analysis and save visualization plots
3. Apply feature engineering to create additional predictive features
4. Train multiple classification models (Random Forest, SVM, KNN, Gradient Boosting, XGBoost, etc.)
5. Create an ensemble model using voting classification
6. Evaluate model performance
7. Select the best model and save it to disk

The enhanced training process employs several techniques to achieve accuracy above 75%:

- Feature engineering to create polynomial, logarithmic, and interaction features
- Advanced algorithms like XGBoost, Gradient Boosting and Ensemble methods
- Extensive hyperparameter tuning
- Cross-validation to prevent overfitting
- Feature importance analysis

### Making Predictions

To predict wine quality using the trained model:

```bash
# Predict a single sample
python predict_wine_quality.py --values <sugar_value> <ph_value>

# Example:
python predict_wine_quality.py --values 10.5 3.2

# Predict multiple samples from a CSV file
python predict_wine_quality.py --csv path/to/your/file.csv

# Use the default dataset (data/wine_quality.csv) for predictions
python predict_wine_quality.py --default
```

The CSV file should contain at least two columns named "Sugar" and "pH".

When using the `--default` option, the script will:

1. Use the training dataset (data/wine_quality.csv) to make predictions
2. If the dataset has a 'Class' column, it will compare predictions with actual values
3. Show prediction accuracy and a summary of correct/incorrect classifications
4. Save the predictions to a new file (data/wine_quality_predictions.csv)

## Model Performance

The enhanced classification model aims to achieve an accuracy above 75% in identifying wine quality.
The actual performance metrics are generated during training and will be displayed when running
the `wine_quality_classifier.py` script.

The model compares two approaches:

1. Basic features (only Sugar and pH)
2. Enhanced features (with engineered features)2. Enhanced features (with engineered features)

For each approach, it trains multiple models including XGBoost (a powerful gradient boosting algorithm) and selects the best one.For each approach, it trains multiple models and selects the best one.

## Visualizations## Visualizations

The following visualizations are generated during the training process:The following visualizations are generated during the training process:

- `wine_quality_eda.png`: Exploratory data analysis plots
- `decision_boundary_<approach>_<model_name>.png`: Decision boundary of each best modeleach best model
- `confusion_matrix_<model_name>.png`: Confusion matrices for each model- `confusion_matrix_<model_name>.png`: Confusion matrices for each model

## Example Results## Example Results

A sample prediction might look like:A sample prediction might look like:
