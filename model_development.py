# model_development.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging

logger = logging.getLogger(__name__)

class MLModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.feature_columns = None
        self.model_performance = {}
        
    def prepare_data(self, df_processed, test_size=0.2):
        """Prepare data for model training"""
        logger.info("Preparing data for model training...")
        
        # Separate features and target
        X = df_processed.drop('attack_detected', axis=1)
        y = df_processed['attack_detected']
        
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        logger.info(f"Features: {len(self.feature_columns)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train, y_train, optimize_hyperparams=True):
        """Train Random Forest model with optional hyperparameter optimization"""
        logger.info("Training Random Forest model...")
        
        if optimize_hyperparams:
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
            
            rf = RandomForestClassifier(random_state=self.random_state)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            best_rf = grid_search.best_estimator_
            
            logger.info(f"Best RF parameters: {grid_search.best_params_}")
            logger.info(f"Best RF CV score: {grid_search.best_score_:.4f}")
            
        else:
            # Use default parameters
            best_rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                bootstrap=True,
                random_state=self.random_state
            )
            best_rf.fit(X_train, y_train)
        
        self.models['RandomForest'] = best_rf
        return best_rf
    
    def train_gradient_boosting(self, X_train, y_train, optimize_hyperparams=True):
        """Train Gradient Boosting model with optional hyperparameter optimization"""
        logger.info("Training Gradient Boosting model...")
        
        if optimize_hyperparams:
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 6, 9],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            gb = GradientBoostingClassifier(random_state=self.random_state)
            grid_search = GridSearchCV(
                gb, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            best_gb = grid_search.best_estimator_
            
            logger.info(f"Best GB parameters: {grid_search.best_params_}")
            logger.info(f"Best GB CV score: {grid_search.best_score_:.4f}")
            
        else:
            # Use default parameters
            best_gb = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            )
            best_gb.fit(X_train, y_train)
        
        self.models['GradientBoosting'] = best_gb
        return best_gb
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Comprehensive model evaluation"""
        logger.info(f"Evaluating {model_name} model...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Store performance metrics
        self.model_performance[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Print results
        print(f"\n{model_name} Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC Score: {auc_score:.4f}")
        
        # Classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Attack', 'Attack'],
                   yticklabels=['No Attack', 'Attack'])
        plt.title(f'{model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return self.model_performance[model_name]
    
    def plot_feature_importance(self, model, model_name, top_n=15):
        """Plot feature importance"""
        if hasattr(model, 'feature_importances_'):
            # Get feature importances
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Plot top N features
            plt.figure(figsize=(10, 8))
            top_features = feature_importance_df.head(top_n)
            sns.barplot(data=top_features, x='importance', y='feature')
            plt.title(f'{model_name} - Top {top_n} Feature Importances')
            plt.xlabel('Feature Importance')
            plt.tight_layout()
            plt.show()
            
            print(f"\nTop {top_n} Most Important Features ({model_name}):")
            for i, (_, row) in enumerate(top_features.iterrows(), 1):
                print(f"{i}. {row['feature']}: {row['importance']:.4f}")
            
            return feature_importance_df
        else:
            logger.warning(f"Model {model_name} does not have feature_importances_ attribute")
            return None
    
    def plot_roc_curves(self, X_test, y_test):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, performance in self.model_performance.items():
            y_pred_proba = performance['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = performance['auc_score']
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    
    def select_best_model(self):
        """Select the best model based on F1-score"""
        if not self.model_performance:
            logger.error("No models have been evaluated yet")
            return None
        
        best_model_name = max(self.model_performance.keys(), 
                             key=lambda x: self.model_performance[x]['f1_score'])
        
        self.best_model = self.models[best_model_name]
        
        logger.info(f"Best model selected: {best_model_name}")
        logger.info(f"Best F1-score: {self.model_performance[best_model_name]['f1_score']:.4f}")
        
        return best_model_name, self.best_model
    
    def save_model(self, model, filepath):
        """Save trained model"""
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def train_and_evaluate_all(self, df_processed, optimize_hyperparams=False):
        """Complete training and evaluation pipeline"""
        logger.info("Starting complete ML pipeline...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df_processed)
        
        # Train models
        rf_model = self.train_random_forest(X_train, y_train, optimize_hyperparams)
        gb_model = self.train_gradient_boosting(X_train, y_train, optimize_hyperparams)
        
        # Evaluate models
        rf_performance = self.evaluate_model(rf_model, X_test, y_test, 'RandomForest')
        gb_performance = self.evaluate_model(gb_model, X_test, y_test, 'GradientBoosting')
        
        # Plot feature importances
        rf_importance = self.plot_feature_importance(rf_model, 'RandomForest')
        gb_importance = self.plot_feature_importance(gb_model, 'GradientBoosting')
        
        # Plot ROC curves
        self.plot_roc_curves(X_test, y_test)
        
        # Select best model
        best_model_name, best_model = self.select_best_model()
        
        # Save best model
        self.save_model(best_model, f'best_model_{best_model_name.lower()}.joblib')
        
        return {
            'best_model_name': best_model_name,
            'best_model': best_model,
            'test_data': (X_test, y_test),
            'feature_importances': {
                'RandomForest': rf_importance,
                'GradientBoosting': gb_importance
            },
            'performance_metrics': self.model_performance
        }

# Usage example
if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    processed_df, original_df = preprocessor.preprocess_pipeline()
    
    # Train and evaluate models
    trainer = MLModelTrainer()
    results = trainer.train_and_evaluate_all(processed_df, optimize_hyperparams=False)
    
    print(f"Best model: {results['best_model_name']}")
    print("Training completed successfully!")
