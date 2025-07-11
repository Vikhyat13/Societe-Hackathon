import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from datetime import datetime, timedelta
import time

# Import our custom modules
from data_preprocessing import DataPreprocessor
from model_development import MLModelTrainer
from risk_scoring import RiskScoringSystem
from root_cause_analysis import RootCauseAnalyzer

class SecurityDashboard:
    def __init__(self):
        self.preprocessor = None
        self.rca_analyzer = None

    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'processed_alerts' not in st.session_state:
            st.session_state.processed_alerts = None
        if 'feedback_data' not in st.session_state:
            st.session_state.feedback_data = []
        if 'suppression_threshold' not in st.session_state:
            st.session_state.suppression_threshold = 20
        if 'risk_system' not in st.session_state:
            st.session_state.risk_system = None

    def load_data_and_model(self):
        """Load data and trained model"""
        try:
            self.preprocessor = DataPreprocessor()
            processed_df, original_df = self.preprocessor.preprocess_pipeline()

            # Initialize risk scoring system and store in session state
            risk_system = RiskScoringSystem(threshold=st.session_state.suppression_threshold)
            try:
                risk_system.load_model('best_model_randomforest.joblib')
                st.session_state.model_loaded = True
            except Exception:
                st.warning("No pre-trained model found. Training new model...")
                trainer = MLModelTrainer()
                results = trainer.train_and_evaluate_all(processed_df, optimize_hyperparams=False)
                risk_system.model = results['best_model']
                st.session_state.model_loaded = True
                st.success("Model trained successfully!")

            st.session_state.risk_system = risk_system
            st.session_state.processed_df = processed_df
            st.session_state.original_df = original_df
            st.session_state.data_loaded = True
            return True
        except Exception as e:
            st.error(f"Error loading data and model: {str(e)}")
            return False

    def process_alerts(self):
        """Process alerts using the risk scoring system"""
        if not st.session_state.data_loaded or not st.session_state.model_loaded:
            st.warning("Please load data and model first.")
            return None

        risk_system = st.session_state.get('risk_system', None)
        if risk_system is None:
            st.error("Risk scoring system is not initialized. Please load data and model first.")
            return None

        processed_df = st.session_state.processed_df
        original_df = st.session_state.original_df

        X = processed_df.drop('attack_detected', axis=1)
        risk_system.update_threshold(st.session_state.suppression_threshold)

        alert_results = risk_system.process_alerts(
            features_df=X,
            session_ids=original_df['session_id'].tolist(),
            additional_features=original_df
        )

        alert_results['actual_attacks'] = processed_df['attack_detected'].values
        st.session_state.processed_alerts = alert_results

        return alert_results

    def render_header(self):
        st.set_page_config(
            page_title="IT Security Operations Dashboard",
            page_icon="🛡️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.title("🛡️ Intelligent IT Security Operations Dashboard")
        st.markdown("### Reducing False Positives with Machine Learning")
        col1, col2, col3 = st.columns(3)
	
        with col1:
            if st.session_state.data_loaded:
                st.success("✅ Data Loaded")
            else:
                st.error("❌ Data Not Loaded")

        with col2:
            if st.session_state.model_loaded:
                st.success("✅ Model Ready")
            else:
                st.error("❌ Model Not Ready")

        with col3:
            if st.session_state.processed_alerts is not None:
                st.success("✅ Alerts Processed")
            else:
                st.warning("⏳ Alerts Pending")

    def render_sidebar(self):
        st.sidebar.header("🔧 Controls")
        if st.sidebar.button("🔄 Load Data & Model", type="primary"):
            with st.spinner("Loading data and model..."):
                self.load_data_and_model()
        if st.session_state.data_loaded and st.session_state.model_loaded:
            st.sidebar.subheader("Alert Processing")
            new_threshold = st.sidebar.slider(
                "Suppression Threshold",
                min_value=0,
                max_value=100,
                value=st.session_state.suppression_threshold,
                help="Alerts below this risk score will be suppressed"
            )
            if new_threshold != st.session_state.suppression_threshold:
                st.session_state.suppression_threshold = new_threshold
            if st.sidebar.button("🚨 Process Alerts"):
                with st.spinner("Processing alerts..."):
                    self.process_alerts()
                st.sidebar.success("Alerts processed!")
        st.sidebar.subheader("Auto-Refresh")
        auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh (30s)")
        if auto_refresh:
            time.sleep(30)
            st.rerun()

    def render_overview_metrics(self):
        if not st.session_state.processed_alerts:
            st.info("Process alerts to see metrics")
            return
        alert_results = st.session_state.processed_alerts
        total_alerts = len(alert_results['session_ids'])
        suppressed_count = alert_results['suppressed'].sum()
        surfaced_count = alert_results['surfaced'].sum()
        avg_risk_score = alert_results['risk_scores'].mean()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Alerts", total_alerts)
        with col2:
            st.metric("Suppressed Alerts", suppressed_count, f"{(suppressed_count/total_alerts)*100:.1f}%")
        with col3:
            st.metric("Surfaced Alerts", surfaced_count, f"{(surfaced_count/total_alerts)*100:.1f}%")
        with col4:
            st.metric("Avg Risk Score", f"{avg_risk_score:.1f}")

    def render_alert_distribution(self):
        if not st.session_state.processed_alerts:
            return
        alert_results = st.session_state.processed_alerts
        col1, col2 = st.columns(2)
        with col1:
            suppressed_count = alert_results['suppressed'].sum()
            surfaced_count = alert_results['surfaced'].sum()
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Suppressed', 'Surfaced'],
                values=[suppressed_count, surfaced_count],
                hole=0.4,
                marker_colors=['#ff9999', '#66b3ff']
            )])
            fig_pie.update_layout(title="Alert Distribution", showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            risk_scores = alert_results['risk_scores']
            suppressed_mask = alert_results['suppressed']
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=risk_scores[suppressed_mask],
                name='Suppressed',
                marker_color='#ff9999',
                opacity=0.7,
                nbinsx=20
            ))
            fig_hist.add_trace(go.Histogram(
                x=risk_scores[~suppressed_mask],
                name='Surfaced',
                marker_color='#66b3ff',
                opacity=0.7,
                nbinsx=20
            ))
            fig_hist.add_vline(
                x=st.session_state.suppression_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text="Threshold"
            )
            fig_hist.update_layout(
                title="Risk Score Distribution",
                xaxis_title="Risk Score",
                yaxis_title="Count",
                barmode='overlay'
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    def render_model_performance(self):
        if not st.session_state.processed_alerts:
            return
        alert_results = st.session_state.processed_alerts
        actual_attacks = alert_results['actual_attacks']
        predicted_surface = alert_results['surfaced']
        tp = ((actual_attacks == 1) & (predicted_surface == True)).sum()
        fp = ((actual_attacks == 0) & (predicted_surface == True)).sum()
        tn = ((actual_attacks == 0) & (predicted_surface == False)).sum()
        fn = ((actual_attacks == 1) & (predicted_surface == False)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Performance Metrics")
            metrics_df = pd.DataFrame({
                'Metric': ['Precision', 'Recall', 'F1-Score', 'Accuracy'],
                'Value': [precision, recall, f1_score, accuracy]
            })
            fig_metrics = px.bar(
                metrics_df,
                x='Metric',
                y='Value',
                title="Model Performance at Current Threshold"
            )
            fig_metrics.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig_metrics, use_container_width=True)
        with col2:
            st.subheader("🎯 Confusion Matrix")
            confusion_matrix = np.array([[tn, fp], [fn, tp]])
            fig_cm = px.imshow(
                confusion_matrix,
                text_auto=True,
                aspect="auto",
                title="Confusion Matrix",
                labels=dict(x="Predicted", y="Actual"),
                x=['Suppressed', 'Surfaced'],
                y=['No Attack', 'Attack']
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    def render_suppression_details(self):
        if not st.session_state.processed_alerts:
            return
        alert_results = st.session_state.processed_alerts
        original_df = st.session_state.original_df
        st.subheader("🔍 Alert Details")
        alert_details = pd.DataFrame({
            'Session ID': alert_results['session_ids'],
            'Risk Score': alert_results['risk_scores'],
            'Status': ['Suppressed' if supp else 'Surfaced' for supp in alert_results['suppressed']],
            'Actual Attack': alert_results['actual_attacks'],
            'Protocol': original_df['protocol_type'].values,
            'Browser': original_df['browser_type'].values,
            'Failed Logins': original_df['failed_logins'].values,
            'IP Reputation': original_df['ip_reputation_score'].values
        })
        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.selectbox("Filter by Status", ["All", "Suppressed", "Surfaced"])
        with col2:
            min_risk = st.number_input("Min Risk Score", min_value=0.0, max_value=100.0, value=0.0)
        with col3:
            max_risk = st.number_input("Max Risk Score", min_value=0.0, max_value=100.0, value=100.0)
        filtered_df = alert_details[
            (alert_details['Risk Score'] >= min_risk) &
            (alert_details['Risk Score'] <= max_risk)
        ]
        if status_filter != "All":
            filtered_df = filtered_df[filtered_df['Status'] == status_filter]
        st.dataframe(filtered_df, use_container_width=True)
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Alert Details",
            data=csv,
            file_name=f"alert_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    def render_analyst_feedback(self):
        st.subheader("📝 Analyst Feedback")
        if not st.session_state.processed_alerts:
            st.info("Process alerts to provide feedback")
            return
        alert_results = st.session_state.processed_alerts
        with st.form("feedback_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                session_id = st.selectbox(
                    "Select Session ID",
                    options=alert_results['session_ids'][:50]
                )
            with col2:
                feedback_type = st.selectbox(
                    "Feedback Type",
                    options=["Correct Suppression", "Incorrect Suppression", "Correct Surface", "Incorrect Surface"]
                )
            with col3:
                confidence = st.slider("Confidence Level", 1, 5, 3)
            comments = st.text_area("Additional Comments")
            submitted = st.form_submit_button("Submit Feedback")
            if submitted:
                feedback_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'session_id': session_id,
                    'feedback_type': feedback_type,
                    'confidence': confidence,
                    'comments': comments
                }
                st.session_state.feedback_data.append(feedback_entry)
                st.success("Feedback submitted successfully!")
        if st.session_state.feedback_data:
            st.subheader("Recent Feedback")
            feedback_df = pd.DataFrame(st.session_state.feedback_data)
            st.dataframe(feedback_df.tail(10), use_container_width=True)

    def render_root_cause_analysis(self):
        st.subheader("🔬 Root Cause Analysis")
        if not st.session_state.processed_alerts:
            st.info("Process alerts to perform root cause analysis")
            return
        if st.button("🚀 Perform Root Cause Analysis"):
            with st.spinner("Analyzing suppressed alerts patterns..."):
                try:
                    rca_analyzer = RootCauseAnalyzer()
                    alert_results = st.session_state.processed_alerts
                    original_df = st.session_state.original_df
                    rca_results = rca_analyzer.perform_complete_analysis(
                        original_df=original_df,
                        suppressed_mask=alert_results['suppressed'],
                        risk_scores=alert_results['risk_scores']
                    )
                    if rca_results:
                        st.success("Root cause analysis completed!")
                        st.subheader("Key Findings")
                        kmeans_analysis = rca_results['clustering_results']['kmeans']['analysis']
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Identified Patterns", len(kmeans_analysis))
                        with col2:
                            total_suppressed = sum(cluster['size'] for cluster in kmeans_analysis.values())
                            st.metric("Analyzed Alerts", total_suppressed)
                        for cluster_id, cluster_info in kmeans_analysis.items():
                            with st.expander(f"Pattern {cluster_id} ({cluster_info['size']} alerts)"):
                                st.write(f"**Average Risk Score:** {cluster_info['avg_risk_score']:.2f}")
                                st.write("**Common Characteristics:**")
                                for char in cluster_info['common_characteristics']:
                                    st.write(f"- {char}")
                        fp_sources = rca_results['false_positive_sources']['kmeans']
                        if fp_sources.get('recommendations'):
                            st.subheader("🎯 Recommendations")
                            for i, rec in enumerate(fp_sources['recommendations'], 1):
                                st.write(f"{i}. {rec['description']} (Priority: {rec['priority']})")
                except Exception as e:
                    st.error(f"Error performing root cause analysis: {str(e)}")

    def run(self):
        self.initialize_session_state()
        self.render_header()
        self.render_sidebar()
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview",
            "🎯 Performance",
            "🔍 Alert Details",
            "📝 Feedback",
            "🔬 Root Cause"
        ])
        with tab1:
            st.header("System Overview")
            self.render_overview_metrics()
            st.divider()
            self.render_alert_distribution()
        with tab2:
            st.header("Model Performance")
            self.render_model_performance()
        with tab3:
            st.header("Alert Details")
            self.render_suppression_details()
        with tab4:
            st.header("Analyst Feedback")
            self.render_analyst_feedback()
        with tab5:
            st.header("Root Cause Analysis")
            self.render_root_cause_analysis()

if __name__ == "__main__":
    dashboard = SecurityDashboard()
    dashboard.run()
