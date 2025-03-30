import joblib
import numpy as np
import argparse
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

def load_model_and_scaler():
    # Load the trained model and scaler
    try:
        model = joblib.load('wine_quality_classifier_model.joblib')
        scaler = joblib.load('wine_quality_scaler.joblib')
        
        # Try to load feature names
        try:
            with open('feature_names.txt', 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            feature_names = ['Sugar', 'pH']
            print("Feature names file not found. Using default features: Sugar, pH")
        
        return model, scaler, feature_names
    except FileNotFoundError:
        print("Error: Model files not found. Please run wine_quality_classifier.py first to train the model.")
        return None, None, None

def engineer_features_for_prediction(data, feature_names):
    """Apply the same feature engineering as during training"""
    # Create a copy to avoid modifying the original
    engineered_data = data.copy()
    
    # Check if we need to add engineered features
    if 'Sugar_Squared' in feature_names and 'Sugar' in data.columns and 'pH' in data.columns:
        # Add polynomial features
        engineered_data['Sugar_Squared'] = engineered_data['Sugar'] ** 2
        engineered_data['pH_Squared'] = engineered_data['pH'] ** 2
        engineered_data['Sugar_pH_Interaction'] = engineered_data['Sugar'] * engineered_data['pH']
        
        # Add logarithmic features
        engineered_data['Log_Sugar'] = np.log1p(engineered_data['Sugar'])
        
        # Add ratio feature
        engineered_data['Sugar_to_pH_Ratio'] = engineered_data['Sugar'] / engineered_data['pH']
        
        # Binning features - need to use consistent bin edges from training
        # For simplicity, we'll use quartile-based binning which should be adaptable
        try:
            # Create bins
            engineered_data['Sugar_Bin'] = pd.qcut(engineered_data['Sugar'], 4, labels=False)
            engineered_data['pH_Bin'] = pd.qcut(engineered_data['pH'], 4, labels=False)
            
            # Convert to one-hot encoding
            sugar_bins = pd.get_dummies(engineered_data['Sugar_Bin'], prefix='Sugar_Bin')
            ph_bins = pd.get_dummies(engineered_data['pH_Bin'], prefix='pH_Bin')
            
            # Combine with original dataframe
            engineered_data = pd.concat([engineered_data, sugar_bins, ph_bins], axis=1)
            
            # Drop the intermediate bin columns
            engineered_data.drop(['Sugar_Bin', 'pH_Bin'], axis=1, inplace=True)
        except Exception as e:
            print(f"Warning: Error during binning: {e}")
            print("Some features may be missing from the prediction.")
    
    # Select only the columns that are in feature_names
    # For new data, some one-hot encoded columns might be missing
    # We'll add them with zeros
    for feature in feature_names:
        if feature not in engineered_data.columns:
            engineered_data[feature] = 0
    
    # Ensure correct column order
    return engineered_data[feature_names]

def predict_single_sample(sugar, pH, model, scaler, feature_names):
    # Create a dataframe for the single sample
    sample_df = pd.DataFrame({'Sugar': [sugar], 'pH': [pH]})
    
    # Apply feature engineering
    engineered_sample = engineer_features_for_prediction(sample_df, feature_names)
    
    # Scale the features
    scaled_features = scaler.transform(engineered_sample)
    
    # Make prediction
    prediction = model.predict(scaled_features)[0]
    
    # Get probability if available
    try:
        probability = model.predict_proba(scaled_features)[0]
        # Determine the class and confidence
        quality = 'Good' if prediction == 1 else 'Poor'
        confidence = probability[1] if prediction == 1 else probability[0]
    except:
        quality = 'Good' if prediction == 1 else 'Poor'
        confidence = None
    
    return quality, confidence

def predict_batch_from_csv(file_path, model, scaler, feature_names):
    try:
        # Load the CSV file
        data = pd.read_csv(file_path)
        
        # Check if required columns exist
        if 'Sugar' not in data.columns or 'pH' not in data.columns:
            print("Error: CSV file must contain 'Sugar' and 'pH' columns.")
            return None
        
        # Apply feature engineering
        engineered_data = engineer_features_for_prediction(data, feature_names)
        
        # Scale features
        scaled_features = scaler.transform(engineered_data)
        
        # Make predictions
        predictions = model.predict(scaled_features)
        
        # Add predictions to the original dataframe
        data['Predicted_Quality'] = ['Good' if p == 1 else 'Poor' for p in predictions]
        
        # Add confidence scores if possible
        try:
            probabilities = model.predict_proba(scaled_features)
            confidence_values = []
            for i, pred in enumerate(predictions):
                confidence_values.append(probabilities[i][1] if pred == 1 else probabilities[i][0])
            data['Confidence'] = confidence_values
        except:
            print("Note: This model doesn't provide prediction probabilities.")
        
        # Save results to a new CSV file
        output_file = file_path.replace('.csv', '_predictions.csv')
        data.to_csv(output_file, index=False)
        
        print(f"Predictions saved to {output_file}")
        return data
        
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return None

def predict_default_dataset(model, scaler, feature_names):
    # Default dataset path
    default_path = "data/wine_quality.csv"
    
    # Check if the file exists
    if not os.path.exists(default_path):
        print(f"Error: Default dataset not found at {default_path}")
        return None
    
    try:
        # Load the dataset
        data = pd.read_csv(default_path)
        print(f"Loaded default dataset with {len(data)} samples.")
        
        if 'Class' in data.columns:
            print("Note: This dataset already contains actual class labels ('Class' column).")
            print("Predictions will be compared with actual values.")
            
            # Store original class values
            original_class = data['Class'].copy()
            
            # Make predictions
            result_data = predict_batch_from_csv(default_path, model, scaler, feature_names)
            
            if result_data is not None:
                # Calculate accuracy if result data is available
                correct_predictions = (result_data['Predicted_Quality'] == original_class).sum()
                accuracy = correct_predictions / len(original_class)
                print(f"Prediction accuracy: {accuracy:.4f} ({correct_predictions}/{len(original_class)} correct)")
                
                # Show confusion matrix-like information
                good_as_good = ((original_class == 'Good') & (result_data['Predicted_Quality'] == 'Good')).sum()
                good_as_poor = ((original_class == 'Good') & (result_data['Predicted_Quality'] == 'Poor')).sum()
                poor_as_good = ((original_class == 'Poor') & (result_data['Predicted_Quality'] == 'Good')).sum()
                poor_as_poor = ((original_class == 'Poor') & (result_data['Predicted_Quality'] == 'Poor')).sum()
                
                print("\nPrediction Results:")
                print(f"Good as Good: {good_as_good}")
                print(f"Good as Poor: {good_as_poor}")
                print(f"Poor as Good: {poor_as_good}")
                print(f"Poor as Poor: {poor_as_poor}")
        else:
            # Just make predictions
            predict_batch_from_csv(default_path, model, scaler, feature_names)
        
    except Exception as e:
        print(f"Error processing default dataset: {e}")

def main():
    parser = argparse.ArgumentParser(description='Predict wine quality based on Sugar and pH values')
    
    # Create mutual exclusive group for input methods
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--values', nargs=2, type=float, metavar=('SUGAR', 'PH'),
                        help='Sugar and pH values for prediction')
    input_group.add_argument('--csv', type=str, help='Path to CSV file with Sugar and pH columns')
    input_group.add_argument('--default', action='store_true', 
                        help='Use the default dataset from data/wine_quality.csv')
    
    args = parser.parse_args()
    
    # Load model and scaler
    model, scaler, feature_names = load_model_and_scaler()
    if model is None or scaler is None:
        return
    
    if args.values:
        sugar, pH = args.values
        quality, confidence = predict_single_sample(sugar, pH, model, scaler, feature_names)
        print(f"\nPrediction for Wine with Sugar={sugar} and pH={pH}:")
        print(f"Predicted Quality: {quality}")
        if confidence is not None:
            print(f"Confidence: {confidence:.2f}")
        
    elif args.csv:
        predict_batch_from_csv(args.csv, model, scaler, feature_names)
        
    elif args.default:
        predict_default_dataset(model, scaler, feature_names)

if __name__ == "__main__":
    main()
