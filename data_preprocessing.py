# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        
    def load_and_clean_data(self, filepath='intrusion_data.csv'):
        """Load and perform initial cleaning of the dataset"""
        logger.info("Loading dataset...")
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.any():
                logger.warning(f"Missing values found:\n{missing_values[missing_values > 0]}")
                
                # Handle missing values
                # For numerical columns, use median
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                for col in numerical_cols:
                    if df[col].isnull().any():
                        median_val = df[col].median()
                        df[col].fillna(median_val, inplace=True)
                        logger.info(f"Filled missing values in {col} with median: {median_val}")
                
                # For categorical columns, use mode
                categorical_cols = df.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    if df[col].isnull().any():
                        mode_val = df[col].mode()[0]
                        df[col] = df[col].fillna(mode_val)
                        logger.info(f"Filled missing values in {col} with mode: {mode_val}")
            
            # Remove duplicates
            initial_rows = len(df)
            df.drop_duplicates(inplace=True)
            final_rows = len(df)
            if initial_rows != final_rows:
                logger.info(f"Removed {initial_rows - final_rows} duplicate rows")
            
            # Basic data validation
            if 'attack_detected' not in df.columns:
                raise ValueError("Target column 'attack_detected' not found in dataset")
                
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def encode_categorical_features(self, df):
        """Encode categorical features using Label Encoding"""
        logger.info("Encoding categorical features...")
        
        categorical_columns = ['protocol_type', 'encryption_used', 'browser_type']
        
        for col in categorical_columns:
            if col in df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[f'{col}_encoded'] = self.encoders[col].transform(df[col].astype(str))
                
                logger.info(f"Encoded {col}: {len(self.encoders[col].classes_)} unique categories")
        
        return df
    
    def engineer_features(self, df):
        """Create additional engineered features"""
        logger.info("Engineering additional features...")
        
        # Create risk-related features
        df['failed_login_ratio'] = df['failed_logins'] / (df['login_attempts'] + 1)
        df['session_duration_log'] = np.log1p(df['session_duration'])
        df['packet_size_category'] = pd.cut(df['network_packet_size'], 
                                          bins=[0, 200, 500, 800, float('inf')], 
                                          labels=['small', 'medium', 'large', 'xlarge'])
        df['packet_size_category_encoded'] = LabelEncoder().fit_transform(df['packet_size_category'])
        
        # Interaction features
        df['login_attempts_x_failures'] = df['login_attempts'] * df['failed_logins']
        df['reputation_risk'] = (1 - df['ip_reputation_score']) * df['failed_logins']
        
        logger.info("Feature engineering completed")
        return df
    
    def scale_numerical_features(self, df, fit=True):
        """Scale numerical features using StandardScaler"""
        logger.info("Scaling numerical features...")
        
        numerical_features = [
            'network_packet_size', 'login_attempts', 'session_duration',
            'ip_reputation_score', 'failed_logins', 'failed_login_ratio',
            'session_duration_log', 'login_attempts_x_failures', 'reputation_risk'
        ]
        
        for feature in numerical_features:
            if feature in df.columns:
                if fit:
                    if feature not in self.scalers:
                        self.scalers[feature] = StandardScaler()
                        df[f'{feature}_scaled'] = self.scalers[feature].fit_transform(df[[feature]])
                    else:
                        df[f'{feature}_scaled'] = self.scalers[feature].transform(df[[feature]])
                else:
                    if feature in self.scalers:
                        df[f'{feature}_scaled'] = self.scalers[feature].transform(df[[feature]])
        
        return df
    
    def prepare_features(self, df):
        """Prepare final feature set for modeling"""
        logger.info("Preparing final feature set...")
        
        # Define feature columns
        self.feature_columns = [
            'network_packet_size_scaled', 'login_attempts_scaled', 'session_duration_scaled',
            'ip_reputation_score_scaled', 'failed_logins_scaled', 'unusual_time_access',
            'protocol_type_encoded', 'encryption_used_encoded', 'browser_type_encoded',
            'failed_login_ratio_scaled', 'session_duration_log_scaled',
            'packet_size_category_encoded', 'login_attempts_x_failures_scaled',
            'reputation_risk_scaled'
        ]
        
        # Ensure all feature columns exist
        available_features = [col for col in self.feature_columns if col in df.columns]
        
        if len(available_features) != len(self.feature_columns):
            missing_features = set(self.feature_columns) - set(available_features)
            logger.warning(f"Missing features: {missing_features}")
        
        self.feature_columns = available_features
        logger.info(f"Final feature set: {len(self.feature_columns)} features")
        
        return df[self.feature_columns + ['attack_detected']]
    
    def preprocess_pipeline(self, filepath='intrusion_data.csv'):
        """Complete preprocessing pipeline"""
        logger.info("Starting preprocessing pipeline...")
        
        # Load and clean data
        df = self.load_and_clean_data(filepath)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Scale numerical features
        df = self.scale_numerical_features(df, fit=True)
        
        # Prepare final feature set
        df_processed = self.prepare_features(df)
        
        logger.info("Preprocessing pipeline completed successfully")
        return df_processed, df  # Return both processed and original dataframes

# Usage example
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    processed_df, original_df = preprocessor.preprocess_pipeline()
    print(f"Processed dataset shape: {processed_df.shape}")
    print(f"Features: {preprocessor.feature_columns}")
