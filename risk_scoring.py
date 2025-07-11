# risk_scoring.py
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class RiskScoringSystem:
    def __init__(self, model_path=None, threshold=20, max_score=100):
        """
        Initialize Risk Scoring System
        
        Args:
            model_path: Path to trained model
            threshold: Risk score threshold for suppression (0-100)
            max_score: Maximum risk score (default 100)
        """
        self.model = None
        self.threshold = threshold
        self.max_score = max_score
        self.suppression_log = []
        self.alert_statistics = {
            'total_alerts': 0,
            'surfaced_alerts': 0,
            'suppressed_alerts': 0,
            'correct_suppressions': 0,
            'incorrect_suppressions': 0
        }
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained model for risk scoring"""
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def calculate_risk_score(self, features, use_probability=True):
        """
        Calculate risk score for given features
        
        Args:
            features: Feature array or DataFrame
            use_probability: Use model probability (True) or prediction confidence (False)
        
        Returns:
            Risk scores (0-100)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        try:
            if use_probability:
                # Use probability of positive class
                probabilities = self.model.predict_proba(features)[:, 1]
                risk_scores = probabilities * self.max_score
            else:
                # Use prediction confidence (distance from decision boundary)
                if hasattr(self.model, 'decision_function'):
                    decision_scores = self.model.decision_function(features)
                    # Normalize to 0-100 scale
                    normalized_scores = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min())
                    risk_scores = normalized_scores * self.max_score
                else:
                    # Fallback to probability method
                    probabilities = self.model.predict_proba(features)[:, 1]
                    risk_scores = probabilities * self.max_score
            
            return np.clip(risk_scores, 0, self.max_score)
            
        except Exception as e:
            logger.error(f"Error calculating risk scores: {str(e)}")
            raise
    
    def apply_suppression_rules(self, risk_scores, additional_rules=None):
        """
        Apply suppression rules based on risk scores and additional criteria
        
        Args:
            risk_scores: Array of risk scores
            additional_rules: Dictionary of additional suppression rules
        
        Returns:
            Boolean array indicating which alerts to suppress
        """
        # Basic threshold-based suppression
        suppress_mask = risk_scores < self.threshold
        
        # Apply additional rules if provided
        if additional_rules:
            for rule_name, rule_function in additional_rules.items():
                try:
                    additional_suppress = rule_function()
                    suppress_mask = suppress_mask | additional_suppress
                    logger.info(f"Applied additional rule: {rule_name}")
                except Exception as e:
                    logger.warning(f"Error applying rule {rule_name}: {str(e)}")
        
        return suppress_mask
    
    def process_alerts(self, features_df, session_ids=None, additional_features=None):
        """
        Process alerts and apply risk scoring and suppression
        
        Args:
            features_df: DataFrame with features for risk scoring
            session_ids: List of session IDs for tracking
            additional_features: Additional features for rule-based suppression
        
        Returns:
            Dictionary with processed alerts information
        """
        logger.info(f"Processing {len(features_df)} alerts...")
        
        # Calculate risk scores
        risk_scores = self.calculate_risk_score(features_df)
        
        # Define additional suppression rules
        additional_rules = {}
        
        if additional_features is not None:
            # Rule 1: Suppress low packet size, low duration alerts
            if 'network_packet_size' in additional_features.columns and 'session_duration' in additional_features.columns:
                additional_rules['low_impact'] = lambda: (
                    (additional_features['network_packet_size'] < 100) & 
                    (additional_features['session_duration'] < 60)
                )
            
            # Rule 2: Suppress high reputation IP alerts with low risk
            if 'ip_reputation_score' in additional_features.columns:
                additional_rules['high_reputation'] = lambda: (
                    (additional_features['ip_reputation_score'] > 0.8) & 
                    (risk_scores < self.threshold * 2)
                )
            
            # Rule 3: Suppress single login attempt, no failures
            if 'login_attempts' in additional_features.columns and 'failed_logins' in additional_features.columns:
                additional_rules['single_success_login'] = lambda: (
                    (additional_features['login_attempts'] == 1) & 
                    (additional_features['failed_logins'] == 0) &
                    (risk_scores < self.threshold * 1.5)
                )
        
        # Apply suppression rules
        suppress_mask = self.apply_suppression_rules(risk_scores, additional_rules)
        
        # Create results
        if session_ids is None:
            session_ids = [f"SID_{i:05d}" for i in range(len(features_df))]
        
        results = {
            'session_ids': session_ids,
            'risk_scores': risk_scores,
            'suppressed': suppress_mask,
            'surfaced': ~suppress_mask,
            'suppression_reasons': []
        }
        
        # Track suppression reasons
        for i, suppress in enumerate(suppress_mask):
            reasons = []
            if risk_scores[i] < self.threshold:
                reasons.append(f"Low risk score ({risk_scores[i]:.1f} < {self.threshold})")
            
            # Check additional rules
            for rule_name, rule_function in additional_rules.items():
                try:
                    if rule_function()[i]:
                        reasons.append(f"Rule: {rule_name}")
                except:
                    pass
            
            results['suppression_reasons'].append(reasons)
        
        # Update statistics
        self.alert_statistics['total_alerts'] += len(features_df)
        self.alert_statistics['suppressed_alerts'] += suppress_mask.sum()
        self.alert_statistics['surfaced_alerts'] += (~suppress_mask).sum()
        
        # Log suppression details
        suppression_entry = {
            'timestamp': datetime.now().isoformat(),
            'total_alerts': len(features_df),
            'suppressed_count': suppress_mask.sum(),
            'surfaced_count': (~suppress_mask).sum(),
            'suppression_rate': (suppress_mask.sum() / len(features_df)) * 100,
            'average_risk_score': float(risk_scores.mean()),
            'suppressed_avg_score': float(risk_scores[suppress_mask].mean()) if suppress_mask.any() else 0,
            'surfaced_avg_score': float(risk_scores[~suppress_mask].mean()) if (~suppress_mask).any() else 0
        }
        
        self.suppression_log.append(suppression_entry)
        
        logger.info(f"Alert processing completed:")
        logger.info(f"  - Total alerts: {len(features_df)}")
        logger.info(f"  - Suppressed: {suppress_mask.sum()} ({suppression_entry['suppression_rate']:.1f}%)")
        logger.info(f"  - Surfaced: {(~suppress_mask).sum()} ({100-suppression_entry['suppression_rate']:.1f}%)")
        logger.info(f"  - Average risk score: {suppression_entry['average_risk_score']:.2f}")
        
        return results
    
    def update_threshold(self, new_threshold):
        """Update suppression threshold"""
        old_threshold = self.threshold
        self.threshold = new_threshold
        logger.info(f"Suppression threshold updated from {old_threshold} to {new_threshold}")
    
    def add_analyst_feedback(self, session_id, actual_attack, feedback_type='threshold'):
        """
        Add analyst feedback for model improvement
        
        Args:
            session_id: Session ID that was reviewed
            actual_attack: True if it was actually an attack, False otherwise
            feedback_type: Type of feedback ('threshold', 'false_positive', 'false_negative')
        """
        # Find the session in suppression log
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'actual_attack': actual_attack,
            'feedback_type': feedback_type
        }
        
        # Update statistics based on feedback
        if feedback_type == 'false_positive' and not actual_attack:
            self.alert_statistics['correct_suppressions'] += 1
        elif feedback_type == 'false_negative' and actual_attack:
            self.alert_statistics['incorrect_suppressions'] += 1
        
        logger.info(f"Feedback added for session {session_id}: {feedback_type}")
        
        return feedback_entry
    
    def get_suppression_statistics(self):
        """Get comprehensive suppression statistics"""
        stats = self.alert_statistics.copy()
        
        if stats['total_alerts'] > 0:
            stats['suppression_rate'] = (stats['suppressed_alerts'] / stats['total_alerts']) * 100
            stats['surface_rate'] = (stats['surfaced_alerts'] / stats['total_alerts']) * 100
        else:
            stats['suppression_rate'] = 0
            stats['surface_rate'] = 0
        
        total_feedback = stats['correct_suppressions'] + stats['incorrect_suppressions']
        if stats['suppressed_alerts'] > 0 and total_feedback > 0:
            stats['suppression_accuracy'] = (stats['correct_suppressions'] / total_feedback) * 100
        else:
            stats['suppression_accuracy'] = 0

        
        return stats
    
    def export_suppression_log(self, filepath):
        """Export suppression log to JSON file"""
        try:
            # Convert numpy int64 to int for JSON serialization
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                raise TypeError

            with open(filepath, 'w') as f:
                json.dump(self.suppression_log, f, indent=2, default=convert_types)
            logger.info(f"Suppression log exported to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting suppression log: {str(e)}")
    
    def generate_suppression_report(self):
        """Generate comprehensive suppression report"""
        stats = self.get_suppression_statistics()
        
        print("ALERT SUPPRESSION SYSTEM REPORT")
        print("=" * 40)
        print(f"Suppression Threshold: {self.threshold}")
        print(f"Total Alerts Processed: {stats['total_alerts']}")
        print(f"Alerts Suppressed: {stats['suppressed_alerts']} ({stats['suppression_rate']:.1f}%)")
        print(f"Alerts Surfaced: {stats['surfaced_alerts']} ({stats['surface_rate']:.1f}%)")
        
        if len(self.suppression_log) > 0:
            recent_entry = self.suppression_log[-1]
            print(f"\nRecent Processing Stats:")
            print(f"Average Risk Score: {recent_entry['average_risk_score']:.2f}")
            print(f"Suppressed Avg Score: {recent_entry['suppressed_avg_score']:.2f}")
            print(f"Surfaced Avg Score: {recent_entry['surfaced_avg_score']:.2f}")
        
        if stats['correct_suppressions'] + stats['incorrect_suppressions'] > 0:
            print(f"\nSuppression Accuracy: {stats['suppression_accuracy']:.1f}%")
            print(f"Correct Suppressions: {stats['correct_suppressions']}")
            print(f"Incorrect Suppressions: {stats['incorrect_suppressions']}")
        
        return stats

# Usage example
if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    from model_development import MLModelTrainer
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    processed_df, original_df = preprocessor.preprocess_pipeline()
    
    # Assume we have a trained model
    # For demo, we'll train a quick model
    trainer = MLModelTrainer()
    results = trainer.train_and_evaluate_all(processed_df, optimize_hyperparams=False)
    
    # Initialize risk scoring system
    risk_system = RiskScoringSystem(threshold=25)
    risk_system.model = results['best_model']
    
    # Process some alerts (using test data)
    X_test, y_test = results['test_data']
    
    # Get original features for additional rules
    test_indices = X_test.index
    original_test_features = original_df.loc[test_indices]
    
    # Process alerts
    alert_results = risk_system.process_alerts(
        features_df=X_test,
        session_ids=original_test_features['session_id'].tolist(),
        additional_features=original_test_features
    )
    
    # Generate report
    risk_system.generate_suppression_report()
    
    # Example of adding feedback
    # risk_system.add_analyst_feedback('SID_00001', actual_attack=False, feedback_type='false_positive')
