# main.py
"""
Main execution script for the Intelligent IT Operations System
Runs the complete pipeline from data preprocessing to dashboard deployment
"""

import logging
import sys
import os
from datetime import datetime
import argparse
import pandas as pd


# Import all modules
from data_preprocessing import DataPreprocessor
from eda_analysis import EDAAnalyzer
from model_development import MLModelTrainer
from risk_scoring import RiskScoringSystem
from root_cause_analysis import RootCauseAnalyzer

# Configure logging
def setup_logging():
    """Setup logging configuration"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = f"security_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized")
    return logger

def run_complete_pipeline(data_file='intrusion_data.csv', optimize_models=False, 
                         perform_rca=True, threshold=20):
    """
    Run the complete intelligent IT operations pipeline
    
    Args:
        data_file: Path to the intrusion data CSV file
        optimize_models: Whether to perform hyperparameter optimization
        perform_rca: Whether to perform root cause analysis
        threshold: Risk score threshold for alert suppression
    """
    logger = setup_logging()
    logger.info("Starting Intelligent IT Operations System Pipeline")
    
    try:
        # Step 1: Data Preprocessing
        logger.info("=" * 50)
        logger.info("STEP 1: DATA PREPROCESSING")
        logger.info("=" * 50)
        
        preprocessor = DataPreprocessor()
        processed_df, original_df = preprocessor.preprocess_pipeline(data_file)
        
        logger.info(f"Data preprocessing completed successfully")
        logger.info(f"Original data shape: {original_df.shape}")
        logger.info(f"Processed data shape: {processed_df.shape}")
        
        # Step 2: Exploratory Data Analysis
        logger.info("=" * 50)
        logger.info("STEP 2: EXPLORATORY DATA ANALYSIS")
        logger.info("=" * 50)
        
        eda_analyzer = EDAAnalyzer(original_df, processed_df)
        eda_report = eda_analyzer.generate_comprehensive_report()
        
        logger.info("EDA completed successfully")
        
        # Step 3: Model Development
        logger.info("=" * 50)
        logger.info("STEP 3: MODEL DEVELOPMENT")
        logger.info("=" * 50)
        
        trainer = MLModelTrainer()
        model_results = trainer.train_and_evaluate_all(
            processed_df, 
            optimize_hyperparams=optimize_models
        )
        
        logger.info(f"Best model: {model_results['best_model_name']}")
        logger.info("Model training completed successfully")
        
        # Step 4: Risk Scoring and Alert Suppression
        logger.info("=" * 50)
        logger.info("STEP 4: RISK SCORING AND ALERT SUPPRESSION")
        logger.info("=" * 50)
        
        risk_system = RiskScoringSystem(threshold=threshold)
        risk_system.model = model_results['best_model']
        
        # Process alerts on test data
        X_test, y_test = model_results['test_data']
        test_indices = X_test.index
        original_test_features = original_df.loc[test_indices]
        
        alert_results = risk_system.process_alerts(
            features_df=X_test,
            session_ids=original_test_features['session_id'].tolist(),
            additional_features=original_test_features
        )
        
        # Generate suppression report
        suppression_stats = risk_system.generate_suppression_report()
        
        logger.info("Risk scoring and alert suppression completed")
        
        # Step 5: Root Cause Analysis (optional)
        if perform_rca:
            logger.info("=" * 50)
            logger.info("STEP 5: ROOT CAUSE ANALYSIS")
            logger.info("=" * 50)
            
            rca_analyzer = RootCauseAnalyzer()
            
            # Use full dataset for RCA
            X_full = processed_df.drop('attack_detected', axis=1)
            full_risk_scores = risk_system.calculate_risk_score(X_full)
            full_suppression_mask = risk_system.apply_suppression_rules(full_risk_scores)
            
            rca_results = rca_analyzer.perform_complete_analysis(
                original_df=original_df,
                suppressed_mask=full_suppression_mask,
                risk_scores=full_risk_scores
            )
            
            logger.info("Root cause analysis completed")
        
        # Step 6: Export Results
        logger.info("=" * 50)
        logger.info("STEP 6: EXPORTING RESULTS")
        logger.info("=" * 50)
        
        # Create reports directory
        reports_dir = "reports"
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export suppression log
        risk_system.export_suppression_log(
            os.path.join(reports_dir, f"suppression_log_{timestamp}.json")
        )
        
        # Export processed alerts
        alert_details_df = pd.DataFrame({
            'session_id': alert_results['session_ids'],
            'risk_score': alert_results['risk_scores'],
            'suppressed': alert_results['suppressed'],
            'suppression_reasons': [', '.join(reasons) for reasons in alert_results['suppression_reasons']]
        })
        
        alert_details_df.to_csv(
            os.path.join(reports_dir, f"alert_details_{timestamp}.csv"),
            index=False
        )
        
        logger.info("Results exported successfully")
        
        # Step 7: Generate Summary Report
        logger.info("=" * 50)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 50)
        
        logger.info(f"Total alerts processed: {len(alert_results['session_ids'])}")
        logger.info(f"Alerts suppressed: {alert_results['suppressed'].sum()}")
        logger.info(f"Alerts surfaced: {alert_results['surfaced'].sum()}")
        logger.info(f"Suppression rate: {(alert_results['suppressed'].sum() / len(alert_results['session_ids'])) * 100:.1f}%")
        logger.info(f"Average risk score: {alert_results['risk_scores'].mean():.2f}")
        
        logger.info("Pipeline execution completed successfully!")
        
        return {
            'preprocessor': preprocessor,
            'model_results': model_results,
            'risk_system': risk_system,
            'alert_results': alert_results,
            'rca_results': rca_results if perform_rca else None
        }
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Intelligent IT Operations System')
    parser.add_argument('--data', default='intrusion_data.csv', 
                       help='Path to intrusion data CSV file')
    parser.add_argument('--optimize', action='store_true', 
                       help='Perform hyperparameter optimization')
    parser.add_argument('--no-rca', action='store_true', 
                       help='Skip root cause analysis')
    parser.add_argument('--threshold', type=int, default=20, 
                       help='Risk score threshold for suppression')
    parser.add_argument('--dashboard', action='store_true', 
                       help='Launch dashboard after processing')
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"Error: Data file '{args.data}' not found!")
        print("Please ensure the intrusion_data.csv file is in the working directory.")
        sys.exit(1)
    
    # Run pipeline
    try:
        results = run_complete_pipeline(
            data_file=args.data,
            optimize_models=args.optimize,
            perform_rca=not args.no_rca,
            threshold=args.threshold
        )
        
        print("\n" + "="*60)
        print("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📊 Total alerts processed: {len(results['alert_results']['session_ids'])}")
        print(f"🚫 Alerts suppressed: {results['alert_results']['suppressed'].sum()}")
        print(f"🚨 Alerts surfaced: {results['alert_results']['surfaced'].sum()}")
        print(f"📈 Suppression rate: {(results['alert_results']['suppressed'].sum() / len(results['alert_results']['session_ids'])) * 100:.1f}%")
        print("\n📁 Check the 'reports' directory for detailed results")
        print("📝 Check the 'logs' directory for execution logs")
        
        if args.dashboard:
            print("\n🚀 Launching dashboard...")
            os.system("streamlit run dashboard.py")
        else:
            print("\n🖥️  To launch the dashboard, run: streamlit run dashboard.py")
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
