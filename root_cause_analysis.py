# root_cause_analysis.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class RootCauseAnalyzer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.cluster_models = {}
        self.cluster_results = {}
        self.pattern_analysis = {}
        
    def prepare_suppressed_data(self, original_df, suppressed_mask, risk_scores):
        """
        Prepare suppressed alerts data for clustering analysis
        
        Args:
            original_df: Original dataset
            suppressed_mask: Boolean mask indicating suppressed alerts
            risk_scores: Risk scores for all alerts
        
        Returns:
            DataFrame with suppressed alerts and relevant features
        """
        logger.info("Preparing suppressed alerts data for analysis...")
        
        # Filter suppressed alerts
        suppressed_df = original_df[suppressed_mask].copy()
        suppressed_risk_scores = risk_scores[suppressed_mask]
        
        # Add risk scores to the dataframe
        suppressed_df['risk_score'] = suppressed_risk_scores
        
        logger.info(f"Total suppressed alerts: {len(suppressed_df)}")
        
        return suppressed_df
    
    def extract_clustering_features(self, suppressed_df):
        """
        Extract and prepare features for clustering analysis
        
        Args:
            suppressed_df: DataFrame with suppressed alerts
        
        Returns:
            Scaled feature matrix for clustering
        """
        # Select relevant features for clustering
        clustering_features = [
            'network_packet_size', 'login_attempts', 'session_duration',
            'ip_reputation_score', 'failed_logins', 'unusual_time_access',
            'risk_score'
        ]
        
        # Add encoded categorical features
        categorical_features = ['protocol_type', 'encryption_used', 'browser_type']
        
        # Encode categorical features for clustering
        feature_df = suppressed_df[clustering_features].copy()
        
        for cat_feature in categorical_features:
            if cat_feature in suppressed_df.columns:
                # One-hot encode categorical features
                dummies = pd.get_dummies(suppressed_df[cat_feature], prefix=cat_feature)
                feature_df = pd.concat([feature_df, dummies], axis=1)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(feature_df)
        
        logger.info(f"Clustering features prepared: {scaled_features.shape[1]} features")
        
        return scaled_features, feature_df.columns.tolist()
    
    def find_optimal_clusters(self, scaled_features, max_clusters=10):
        """
        Find optimal number of clusters using elbow method and silhouette score
        
        Args:
            scaled_features: Scaled feature matrix
            max_clusters: Maximum number of clusters to test
        
        Returns:
            Optimal number of clusters
        """
        logger.info("Finding optimal number of clusters...")
        
        if len(scaled_features) < max_clusters:
            max_clusters = len(scaled_features) - 1
        
        inertias = []
        silhouette_scores = []
        cluster_range = range(2, min(max_clusters + 1, len(scaled_features)))
        
        for n_clusters in cluster_range:
            # Fit KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_features)
            
            # Calculate metrics
            inertias.append(kmeans.inertia_)
            
            if len(np.unique(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(scaled_features, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            else:
                silhouette_scores.append(0)
        
        # Plot elbow curve and silhouette scores
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Elbow curve
        ax1.plot(cluster_range, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method for Optimal Clusters')
        ax1.grid(True)
        
        # Silhouette scores
        ax2.plot(cluster_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score vs Number of Clusters')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Find optimal number of clusters (highest silhouette score)
        optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
        
        logger.info(f"Optimal number of clusters: {optimal_clusters}")
        logger.info(f"Best silhouette score: {max(silhouette_scores):.3f}")
        
        return optimal_clusters, silhouette_scores
    
    def perform_kmeans_clustering(self, scaled_features, n_clusters):
        """
        Perform K-means clustering on suppressed alerts
        
        Args:
            scaled_features: Scaled feature matrix
            n_clusters: Number of clusters
        
        Returns:
            KMeans model and cluster labels
        """
        logger.info(f"Performing K-means clustering with {n_clusters} clusters...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        self.cluster_models['kmeans'] = kmeans
        
        logger.info("K-means clustering completed")
        
        return kmeans, cluster_labels
    
    def perform_dbscan_clustering(self, scaled_features, eps=0.5, min_samples=5):
        """
        Perform DBSCAN clustering to find density-based patterns
        
        Args:
            scaled_features: Scaled feature matrix
            eps: Maximum distance between samples in a cluster
            min_samples: Minimum samples required to form a cluster
        
        Returns:
            DBSCAN model and cluster labels
        """
        logger.info(f"Performing DBSCAN clustering (eps={eps}, min_samples={min_samples})...")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(scaled_features)
        
        self.cluster_models['dbscan'] = dbscan
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        logger.info(f"DBSCAN clustering completed: {n_clusters} clusters, {n_noise} noise points")
        
        return dbscan, cluster_labels
    
    def analyze_cluster_patterns(self, suppressed_df, cluster_labels, feature_names, method='kmeans'):
        """
        Analyze patterns within each cluster
        
        Args:
            suppressed_df: DataFrame with suppressed alerts
            cluster_labels: Cluster assignments
            feature_names: Names of features used for clustering
            method: Clustering method used
        
        Returns:
            Dictionary with cluster analysis results
        """
        logger.info(f"Analyzing cluster patterns for {method}...")
        
        # Add cluster labels to dataframe
        analysis_df = suppressed_df.copy()
        analysis_df['cluster'] = cluster_labels
        
        cluster_analysis = {}
        unique_clusters = sorted(set(cluster_labels))
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
                
            cluster_data = analysis_df[analysis_df['cluster'] == cluster_id]
            cluster_size = len(cluster_data)
            
            logger.info(f"Analyzing cluster {cluster_id} ({cluster_size} alerts)...")
            
            # Statistical summary
            numerical_features = ['network_packet_size', 'login_attempts', 'session_duration',
                                'ip_reputation_score', 'failed_logins', 'risk_score']
            
            cluster_stats = {}
            for feature in numerical_features:
                if feature in cluster_data.columns:
                    cluster_stats[feature] = {
                        'mean': cluster_data[feature].mean(),
                        'median': cluster_data[feature].median(),
                        'std': cluster_data[feature].std(),
                        'min': cluster_data[feature].min(),
                        'max': cluster_data[feature].max()
                    }
            
            # Categorical feature distributions
            categorical_distributions = {}
            categorical_features = ['protocol_type', 'encryption_used', 'browser_type']
            
            for feature in categorical_features:
                if feature in cluster_data.columns:
                    value_counts = cluster_data[feature].value_counts()
                    categorical_distributions[feature] = value_counts.to_dict()
            
            # Time-based patterns
            time_patterns = {
                'unusual_time_access_rate': cluster_data['unusual_time_access'].mean()
            }
            
            # Common characteristics
            common_characteristics = []
            
            # High-frequency categorical values
            for feature in categorical_features:
                if feature in cluster_data.columns:
                    mode_value = cluster_data[feature].mode()
                    if len(mode_value) > 0:
                        mode_freq = (cluster_data[feature] == mode_value[0]).sum() / len(cluster_data)
                        if mode_freq > 0.7:  # If more than 70% have the same value
                            common_characteristics.append(f"Mostly {feature}: {mode_value[0]} ({mode_freq:.1%})")
            
            # Risk score characteristics
            avg_risk = cluster_data['risk_score'].mean()
            if avg_risk < 10:
                common_characteristics.append("Very low risk scores")
            elif avg_risk < 20:
                common_characteristics.append("Low risk scores")
            
            cluster_analysis[cluster_id] = {
                'size': cluster_size,
                'percentage': (cluster_size / len(analysis_df)) * 100,
                'statistical_summary': cluster_stats,
                'categorical_distributions': categorical_distributions,
                'time_patterns': time_patterns,
                'common_characteristics': common_characteristics,
                'avg_risk_score': avg_risk
            }
        
        self.cluster_results[method] = cluster_analysis
        
        return cluster_analysis
    
    def identify_false_positive_sources(self, cluster_analysis, method='kmeans'):
        """
        Identify common sources and causes of false positives
        
        Args:
            cluster_analysis: Results from cluster analysis
            method: Clustering method used
        
        Returns:
            Dictionary with identified false positive sources
        """
        logger.info("Identifying false positive sources...")
        
        fp_sources = {
            'low_risk_patterns': [],
            'common_configurations': [],
            'temporal_patterns': [],
            'recommendations': []
        }
        
        for cluster_id, cluster_info in cluster_analysis.items():
            cluster_size = cluster_info['size']
            avg_risk = cluster_info['avg_risk_score']
            
            # Low risk clusters with significant size
            if avg_risk < 15 and cluster_size > 10:
                pattern_description = f"Cluster {cluster_id}: {cluster_size} alerts with avg risk {avg_risk:.1f}"
                fp_sources['low_risk_patterns'].append({
                    'cluster_id': cluster_id,
                    'description': pattern_description,
                    'characteristics': cluster_info['common_characteristics']
                })
            
            # Common configurations causing false positives
            for feature, distribution in cluster_info['categorical_distributions'].items():
                dominant_value = max(distribution.items(), key=lambda x: x[1])
                if dominant_value[1] / cluster_size > 0.8:  # More than 80% have same value
                    fp_sources['common_configurations'].append({
                        'feature': feature,
                        'value': dominant_value[0],
                        'frequency': dominant_value[1],
                        'cluster_id': cluster_id,
                        'cluster_size': cluster_size
                    })
            
            # Temporal patterns
            if cluster_info['time_patterns']['unusual_time_access_rate'] > 0.7:
                fp_sources['temporal_patterns'].append({
                    'cluster_id': cluster_id,
                    'pattern': 'High unusual time access',
                    'rate': cluster_info['time_patterns']['unusual_time_access_rate'],
                    'cluster_size': cluster_size
                })
        
        # Generate recommendations
        fp_sources['recommendations'] = self._generate_recommendations(fp_sources)
        
        self.pattern_analysis[method] = fp_sources
        
        return fp_sources
    
    def _generate_recommendations(self, fp_sources):
        """Generate recommendations based on identified patterns"""
        recommendations = []
        
        # Low risk pattern recommendations
        if fp_sources['low_risk_patterns']:
            recommendations.append({
                'type': 'threshold_adjustment',
                'description': 'Consider lowering suppression threshold for very low-risk alerts',
                'affected_clusters': [p['cluster_id'] for p in fp_sources['low_risk_patterns']],
                'priority': 'high'
            })
        
        # Configuration-based recommendations
        config_patterns = {}
        for config in fp_sources['common_configurations']:
            key = f"{config['feature']}_{config['value']}"
            if key not in config_patterns:
                config_patterns[key] = []
            config_patterns[key].append(config)
        
        for pattern_key, configs in config_patterns.items():
            if len(configs) > 1:  # Pattern appears in multiple clusters
                total_affected = sum(c['cluster_size'] for c in configs)
                recommendations.append({
                    'type': 'rule_based_suppression',
                    'description': f"Create specific suppression rule for {pattern_key}",
                    'affected_alerts': total_affected,
                    'priority': 'medium'
                })
        
        # Temporal pattern recommendations
        if fp_sources['temporal_patterns']:
            recommendations.append({
                'type': 'temporal_rules',
                'description': 'Consider time-based suppression rules for unusual access patterns',
                'affected_clusters': [p['cluster_id'] for p in fp_sources['temporal_patterns']],
                'priority': 'low'
            })
        
        return recommendations
    
    def visualize_clusters(self, scaled_features, cluster_labels, method='kmeans'):
        """
        Visualize clusters using PCA for dimensionality reduction
        
        Args:
            scaled_features: Scaled feature matrix
            cluster_labels: Cluster assignments
            method: Clustering method used
        """
        logger.info(f"Visualizing {method} clusters...")
        
        # Apply PCA for visualization
        pca = PCA(n_components=2, random_state=self.random_state)
        features_2d = pca.fit_transform(scaled_features)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        unique_labels = sorted(set(cluster_labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:  # Noise points in DBSCAN
                plt.scatter(features_2d[cluster_labels == label, 0],
                           features_2d[cluster_labels == label, 1],
                           c='black', marker='x', s=50, alpha=0.5, label='Noise')
            else:
                plt.scatter(features_2d[cluster_labels == label, 0],
                           features_2d[cluster_labels == label, 1],
                           c=[color], s=50, alpha=0.7, label=f'Cluster {label}')
        
        plt.xlabel(f'First Principal Component (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'Second Principal Component (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})')
        plt.title(f'{method.upper()} Clustering Results (PCA Visualization)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        logger.info(f"Total explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    def generate_rca_report(self, method='kmeans'):
        """
        Generate comprehensive root cause analysis report
        
        Args:
            method: Clustering method to report on
        
        Returns:
            Formatted report string
        """
        if method not in self.cluster_results:
            logger.error(f"No cluster results found for method: {method}")
            return None
        
        cluster_analysis = self.cluster_results[method]
        fp_sources = self.pattern_analysis.get(method, {})
        
        print(f"ROOT CAUSE ANALYSIS REPORT - {method.upper()}")
        print("=" * 50)
        
        # Cluster summary
        total_clusters = len(cluster_analysis)
        total_alerts = sum(c['size'] for c in cluster_analysis.values())
        
        print(f"Total Clusters Identified: {total_clusters}")
        print(f"Total Suppressed Alerts Analyzed: {total_alerts}")
        
        # Cluster details
        print(f"\nCLUSTER BREAKDOWN:")
        for cluster_id, info in cluster_analysis.items():
            print(f"\nCluster {cluster_id}:")
            print(f"  Size: {info['size']} alerts ({info['percentage']:.1f}%)")
            print(f"  Average Risk Score: {info['avg_risk_score']:.2f}")
            print(f"  Key Characteristics:")
            for char in info['common_characteristics']:
                print(f"    - {char}")
        
        # False positive sources
        if fp_sources:
            print(f"\nIDENTIFIED FALSE POSITIVE SOURCES:")
            
            if fp_sources.get('low_risk_patterns'):
                print(f"\nLow Risk Patterns:")
                for pattern in fp_sources['low_risk_patterns']:
                    print(f"  - {pattern['description']}")
            
            if fp_sources.get('common_configurations'):
                print(f"\nCommon Configurations:")
                config_summary = {}
                for config in fp_sources['common_configurations']:
                    key = f"{config['feature']}={config['value']}"
                    if key not in config_summary:
                        config_summary[key] = 0
                    config_summary[key] += config['cluster_size']
                
                for config, count in sorted(config_summary.items(), key=lambda x: x[1], reverse=True):
                    print(f"  - {config}: {count} alerts")
            
            # Recommendations
            if fp_sources.get('recommendations'):
                print(f"\nRECOMMENDATIONS:")
                for i, rec in enumerate(fp_sources['recommendations'], 1):
                    print(f"{i}. {rec['description']} (Priority: {rec['priority']})")
        
        return cluster_analysis
    
    def perform_complete_analysis(self, original_df, suppressed_mask, risk_scores):
        """
        Perform complete root cause analysis
        
        Args:
            original_df: Original dataset
            suppressed_mask: Boolean mask indicating suppressed alerts
            risk_scores: Risk scores for all alerts
        
        Returns:
            Complete analysis results
        """
        logger.info("Starting complete root cause analysis...")
        
        # Prepare data
        suppressed_df = self.prepare_suppressed_data(original_df, suppressed_mask, risk_scores)
        
        if len(suppressed_df) < 10:
            logger.warning("Too few suppressed alerts for meaningful clustering analysis")
            return None
        
        # Extract features
        scaled_features, feature_names = self.extract_clustering_features(suppressed_df)
        
        # Find optimal clusters
        optimal_k, silhouette_scores = self.find_optimal_clusters(scaled_features)
        
        # Perform clustering
        kmeans_model, kmeans_labels = self.perform_kmeans_clustering(scaled_features, optimal_k)
        dbscan_model, dbscan_labels = self.perform_dbscan_clustering(scaled_features)
        
        # Analyze patterns
        kmeans_analysis = self.analyze_cluster_patterns(suppressed_df, kmeans_labels, feature_names, 'kmeans')
        dbscan_analysis = self.analyze_cluster_patterns(suppressed_df, dbscan_labels, feature_names, 'dbscan')
        
        # Identify false positive sources
        kmeans_sources = self.identify_false_positive_sources(kmeans_analysis, 'kmeans')
        dbscan_sources = self.identify_false_positive_sources(dbscan_analysis, 'dbscan')
        
        # Visualize results
        self.visualize_clusters(scaled_features, kmeans_labels, 'kmeans')
        self.visualize_clusters(scaled_features, dbscan_labels, 'dbscan')
        
        # Generate reports
        print("\n" + "="*80)
        self.generate_rca_report('kmeans')
        print("\n" + "="*80)
        self.generate_rca_report('dbscan')
        
        return {
            'suppressed_data': suppressed_df,
            'scaled_features': scaled_features,
            'feature_names': feature_names,
            'optimal_clusters': optimal_k,
            'clustering_results': {
                'kmeans': {'model': kmeans_model, 'labels': kmeans_labels, 'analysis': kmeans_analysis},
                'dbscan': {'model': dbscan_model, 'labels': dbscan_labels, 'analysis': dbscan_analysis}
            },
            'false_positive_sources': {
                'kmeans': kmeans_sources,
                'dbscan': dbscan_sources
            }
        }

# Usage example
if __name__ == "__main__":
    # This would typically be called after running the risk scoring system
    pass
