import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

class CarbonDataProcessor:
    """Process and prepare carbon footprint data for ML models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load carbon footprint dataset"""
        try:
            df = pd.read_csv(filepath)
            print(f"Dataset loaded: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the dataset"""
        print("Cleaning data...")
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Remove outliers using IQR method
        for col in numeric_columns:
            if col != 'carbon_footprint':  # Don't remove outliers from target
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        print(f"Data after cleaning: {df.shape}")
        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features for better prediction"""
        print("Engineering features...")
        
        # Energy efficiency ratio
        if 'energy_consumption' in df.columns and 'renewable_energy' in df.columns:
            df['energy_efficiency'] = df['renewable_energy'] / (df['energy_consumption'] + 1)
        
        # Transportation intensity
        if 'transport_miles' in df.columns and 'fuel_consumption' in df.columns:
            df['transport_intensity'] = df['fuel_consumption'] / (df['transport_miles'] + 1)
        
        # Industrial emission factor
        if 'industrial_output' in df.columns and 'industrial_emissions' in df.columns:
            df['emission_factor'] = df['industrial_emissions'] / (df['industrial_output'] + 1)
        
        # Waste management efficiency
        if 'waste_generated' in df.columns and 'waste_recycled' in df.columns:
            df['waste_efficiency'] = df['waste_recycled'] / (df['waste_generated'] + 1)
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col])
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, target_column: str = 'carbon_footprint') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for training"""
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y.values
    
    def visualize_data(self, df: pd.DataFrame, target_column: str = 'carbon_footprint'):
        """Create visualizations for data exploration"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Target distribution
        axes[0, 0].hist(df[target_column], bins=30, alpha=0.7)
        axes[0, 0].set_title('Carbon Footprint Distribution')
        axes[0, 0].set_xlabel('Carbon Footprint (tons CO2)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Correlation heatmap
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0, 1])
        axes[0, 1].set_title('Feature Correlation Matrix')
        
        # Feature importance (top 10 features correlated with target)
        target_corr = corr_matrix[target_column].abs().sort_values(ascending=False)[1:11]
        axes[1, 0].barh(range(len(target_corr)), target_corr.values)
        axes[1, 0].set_yticks(range(len(target_corr)))
        axes[1, 0].set_yticklabels(target_corr.index)
        axes[1, 0].set_title('Top 10 Features Correlated with Carbon Footprint')
        axes[1, 0].set_xlabel('Absolute Correlation')
        
        # Scatter plot of top correlated feature
        top_feature = target_corr.index[0]
        axes[1, 1].scatter(df[top_feature], df[target_column], alpha=0.6)
        axes[1, 1].set_xlabel(top_feature)
        axes[1, 1].set_ylabel('Carbon Footprint')
        axes[1, 1].set_title(f'Carbon Footprint vs {top_feature}')
        
        plt.tight_layout()
        plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Create sample dataset for demonstration"""
        np.random.seed(42)
        
        data = {
            'energy_consumption': np.random.normal(5000, 1500, n_samples),
            'renewable_energy': np.random.normal(1000, 500, n_samples),
            'transport_miles': np.random.normal(12000, 4000, n_samples),
            'fuel_consumption': np.random.normal(800, 200, n_samples),
            'industrial_output': np.random.normal(10000, 3000, n_samples),
            'industrial_emissions': np.random.normal(2000, 800, n_samples),
            'waste_generated': np.random.normal(500, 150, n_samples),
            'waste_recycled': np.random.normal(200, 80, n_samples),
            'building_area': np.random.normal(2000, 600, n_samples),
            'heating_type': np.random.choice(['gas', 'electric', 'oil', 'renewable'], n_samples),
            'region': np.random.choice(['urban', 'suburban', 'rural'], n_samples),
            'industry_type': np.random.choice(['manufacturing', 'services', 'technology', 'agriculture'], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable with realistic relationships
        df['carbon_footprint'] = (
            0.3 * df['energy_consumption'] / 1000 +
            -0.1 * df['renewable_energy'] / 1000 +
            0.2 * df['fuel_consumption'] / 100 +
            0.4 * df['industrial_emissions'] / 1000 +
            0.1 * df['waste_generated'] / 100 +
            -0.05 * df['waste_recycled'] / 100 +
            np.random.normal(0, 0.5, n_samples)  # Add noise
        )
        
        # Ensure positive values
        df['carbon_footprint'] = np.maximum(df['carbon_footprint'], 0.1)
        
        return df
