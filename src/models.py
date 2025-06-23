import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class CarbonFootprintPredictor:
    """Carbon footprint prediction using multiple regression algorithms"""
    
    def __init__(self):
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
        self.best_model = None
        self.best_score = float('-inf')
        self.feature_names = []
    
    def train_models(self, X_train, X_test, y_train, y_test, feature_names=None):
        """Train all models and compare performance"""
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        results = {}
        
        print("Training models...")
        print("="*50)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Cross validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred_test
            }
            
            print(f"  R² Score: {test_r2:.4f}")
            print(f"  RMSE: {test_rmse:.4f}")
            print(f"  MAE: {test_mae:.4f}")
            print(f"  CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Update best model
            if test_r2 > self.best_score:
                self.best_score = test_r2
                self.best_model = model
            
            print("-" * 30)
        
        return results, y_test
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for best models"""
        print("Performing hyperparameter tuning...")
        
        # XGBoost tuning
        xgb_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        xgb_grid = GridSearchCV(
            xgb.XGBRegressor(random_state=42),
            xgb_params,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        xgb_grid.fit(X_train, y_train)
        
        # Random Forest tuning
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=42),
            rf_params,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        rf_grid.fit(X_train, y_train)
        
        print(f"Best XGBoost R²: {xgb_grid.best_score_:.4f}")
        print(f"Best XGBoost params: {xgb_grid.best_params_}")
        print(f"Best Random Forest R²: {rf_grid.best_score_:.4f}")
        print(f"Best Random Forest params: {rf_grid.best_params_}")
        
        # Update models with best parameters
        self.models['xgboost'] = xgb_grid.best_estimator_
        self.models['random_forest'] = rf_grid.best_estimator_
        
        return xgb_grid.best_estimator_, rf_grid.best_estimator_
    
    def feature_importance_analysis(self, model_results):
        """Analyze feature importance for tree-based models"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # XGBoost feature importance
        if 'xgboost' in model_results:
            xgb_model = model_results['xgboost']['model']
            xgb_importances = xgb_model.feature_importances_
            
            # Sort features by importance
            indices = np.argsort(xgb_importances)[::-1][:10]
            
            axes[0].bar(range(len(indices)), xgb_importances[indices])
            axes[0].set_title('XGBoost Feature Importance')
            axes[0].set_xlabel('Features')
            axes[0].set_ylabel('Importance')
            axes[0].set_xticks(range(len(indices)))
            axes[0].set_xticklabels([self.feature_names[i] for i in indices], rotation=45)
        
        # Random Forest feature importance
        if 'random_forest' in model_results:
            rf_model = model_results['random_forest']['model']
            rf_importances = rf_model.feature_importances_
            
            # Sort features by importance
            indices = np.argsort(rf_importances)[::-1][:10]
            
            axes[1].bar(range(len(indices)), rf_importances[indices])
            axes[1].set_title('Random Forest Feature Importance')
            axes[1].set_xlabel('Features')
            axes[1].set_ylabel('Importance')
            axes[1].set_xticks(range(len(indices)))
            axes[1].set_xticklabels([self.feature_names[i] for i in indices], rotation=45)
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_results(self, model_results, y_test):
        """Plot model comparison and predictions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Model performance comparison
        models = list(model_results.keys())
        r2_scores = [model_results[model]['test_r2'] for model in models]
        rmse_scores = [model_results[model]['test_rmse'] for model in models]
        
        axes[0, 0].bar(models, r2_scores)
        axes[0, 0].set_title('Model R² Scores')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].bar(models, rmse_scores)
        axes[0, 1].set_title('Model RMSE Scores')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Prediction vs Actual for best model
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['test_r2'])
        best_predictions = model_results[best_model_name]['predictions']
        
        axes[1, 0].scatter(y_test, best_predictions, alpha=0.6)
        axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual Carbon Footprint')
        axes[1, 0].set_ylabel('Predicted Carbon Footprint')
        axes[1, 0].set_title(f'Prediction vs Actual - {best_model_name}')
        
        # Residuals plot
        residuals = y_test - best_predictions
        axes[1, 1].scatter(best_predictions, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Carbon Footprint')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals Plot')
        
        plt.tight_layout()
        plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict(self, features):
        """Make prediction using the best model"""
        if self.best_model is None:
            raise ValueError("No trained model available")
        
        if isinstance(features, list):
            features = np.array(features).reshape(1, -1)
        elif len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        prediction = self.best_model.predict(features)
        return prediction[0] if len(prediction) == 1 else prediction
    
    def save_model(self, filepath):
        """Save the best model"""
        if self.best_model is None:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.best_model,
            'score': self.best_score,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load saved model"""
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.best_score = model_data['score']
        self.feature_names = model_data.get('feature_names', [])
        print(f"Model loaded from {filepath}")
