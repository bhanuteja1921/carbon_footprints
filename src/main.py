import argparse
import json
import numpy as np
import pandas as pd
from data_processor import CarbonDataProcessor
from models import CarbonFootprintPredictor
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description='Carbon Footprint Prediction')
    parser.add_argument('--mode', choices=['train', 'predict', 'demo'], required=True,
                        help='Mode: train, predict, or demo')
    parser.add_argument('--data_path', type=str, help='Path to dataset CSV file')
    parser.add_argument('--input_data', type=str, help='Path to input JSON for prediction')
    parser.add_argument('--model_path', type=str, default='carbon_model.pkl',
                        help='Path to save/load model')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Initialize processor and predictor
    processor = CarbonDataProcessor()
    predictor = CarbonFootprintPredictor()
    
    if args.mode == 'demo':
        print("Running demo with sample data...")
        
        # Create sample dataset
        df = processor.create_sample_data(n_samples=2000)
        print(f"Created sample dataset: {df.shape}")
        
        # Process data
        df_clean = processor.clean_data(df)
        df_engineered = processor.feature_engineering(df_clean)
        df_encoded = processor.encode_categorical(df_engineered)
        
        # Visualize data
        processor.visualize_data(df_encoded)
        
        # Prepare features
        X, y = processor.prepare_features(df_encoded)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Hyperparameter tuning (optional)
        if args.tune:
            predictor.hyperparameter_tuning(X_train, y_train)
        
        # Train models
        results, y_test = predictor.train_models(X_train, X_test, y_train, y_test, 
                                               processor.feature_names)
        
        # Analyze results
        predictor.feature_importance_analysis(results)
        predictor.plot_results(results, y_test)
        
        # Print summary
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        
        for model_name, result in results.items():
            accuracy = result['test_r2'] * 100
            print(f"{model_name:20}: {accuracy:.1f}% accuracy (R² = {result['test_r2']:.4f})")
        
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
        best_accuracy = results[best_model_name]['test_r2'] * 100
        print(f"\nBest Model: {best_model_name} with {best_accuracy:.1f}% accuracy")
        
        # Save best model
        predictor.save_model(args.model_path)
        
        # Demo prediction
        print("\n" + "="*60)
        print("DEMO PREDICTION")
        print("="*60)
        
        # Create sample input
        sample_features = X_test[0]
        actual_value = y_test[0]
        predicted_value = predictor.predict(sample_features)
        
        print(f"Actual carbon footprint: {actual_value:.2f} tons CO2")
        print(f"Predicted carbon footprint: {predicted_value:.2f} tons CO2")
        print(f"Prediction error: {abs(actual_value - predicted_value):.2f} tons CO2")
        print(f"Relative error: {abs(actual_value - predicted_value) / actual_value * 100:.1f}%")
    
    elif args.mode == 'train':
        if not args.data_path:
            print("Data path required for training")
            return
        
        print(f"Loading dataset from {args.data_path}...")
        df = processor.load_data(args.data_path)
        
        if df.empty:
            print("Failed to load dataset")
            return
        
        # Process data
        print("Processing data...")
        df_clean = processor.clean_data(df)
        df_engineered = processor.feature_engineering(df_clean)
        df_encoded = processor.encode_categorical(df_engineered)
        
        # Visualize data
        processor.visualize_data(df_encoded)
        
        # Prepare features
        X, y = processor.prepare_features(df_encoded)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Hyperparameter tuning (optional)
        if args.tune:
            predictor.hyperparameter_tuning(X_train, y_train)
        
        # Train models
        results, y_test = predictor.train_models(X_train, X_test, y_train, y_test,
                                               processor.feature_names)
        
        # Analyze and save results
        predictor.feature_importance_analysis(results)
        predictor.plot_results(results, y_test)
        predictor.save_model(args.model_path)
    
    elif args.mode == 'predict':
        if not args.input_data:
            print("Input data required for prediction")
            return
        
        # Load model
        try:
            predictor.load_model(args.model_path)
        except FileNotFoundError:
            print(f"Model file {args.model_path} not found. Please train first.")
            return
        
        # Load input data
        with open(args.input_data, 'r') as f:
            input_data = json.load(f)
        
        # Convert to features array (order should match training features)
        features = []
        for feature_name in predictor.feature_names:
            if feature_name in input_data:
                features.append(input_data[feature_name])
            else:
                print(f"Warning: Feature '{feature_name}' not found in input data")
                features.append(0)  # Default value
        
        # Make prediction
        prediction = predictor.predict(np.array(features))
        
        print(f"Predicted carbon footprint: {prediction:.2f} tons CO2")
        
        # Create sample input file if it doesn't exist
        if not os.path.exists(args.input_data):
            sample_input = {
                'energy_consumption': 5000,
                'renewable_energy': 1000,
                'transport_miles': 12000,
                'fuel_consumption': 800,
                'industrial_output': 10000,
                'industrial_emissions': 2000,
                'waste_generated': 500,
                'waste_recycled': 200,
                'building_area': 2000
            }
            
            with open('sample_input.json', 'w') as f:
                json.dump(sample_input, f, indent=2)
            
            print(f"Sample input file created: sample_input.json")

if __name__ == "__main__":
    import os
    main()
