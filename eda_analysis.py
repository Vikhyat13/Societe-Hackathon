# eda_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class EDAAnalyzer:
    def __init__(self, df_original, df_processed):
        self.df_original = df_original
        self.df_processed = df_processed
        
        # Set style for better plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def analyze_target_distribution(self):
        """Analyze the distribution of the target variable"""
        print("=== TARGET VARIABLE ANALYSIS ===")
        
        # Count and percentage distribution
        target_counts = self.df_original['attack_detected'].value_counts()
        target_percentages = self.df_original['attack_detected'].value_counts(normalize=True) * 100
        
        print(f"Attack Distribution:")
        print(f"No Attack (0): {target_counts[0]} ({target_percentages[0]:.2f}%)")
        print(f"Attack (1): {target_counts[1]} ({target_percentages[1]:.2f}%)")
        
        # Visualize distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Count plot
        sns.countplot(data=self.df_original, x='attack_detected', ax=ax1)
        ax1.set_title('Attack Detection Distribution')
        ax1.set_xlabel('Attack Detected')
        ax1.set_ylabel('Count')
        
        # Pie chart
        ax2.pie(target_counts.values, labels=['No Attack', 'Attack'], autopct='%1.1f%%')
        ax2.set_title('Attack Detection Percentage')
        
        plt.tight_layout()
        plt.show()
        
        return target_counts, target_percentages
    
    def analyze_numerical_features(self):
        """Analyze numerical feature distributions"""
        print("\n=== NUMERICAL FEATURES ANALYSIS ===")
        
        numerical_cols = ['network_packet_size', 'login_attempts', 'session_duration', 
                         'ip_reputation_score', 'failed_logins']
        
        # Statistical summary
        print("Statistical Summary:")
        print(self.df_original[numerical_cols + ['attack_detected']].groupby('attack_detected').describe())
        
        # Distribution plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(numerical_cols):
            # Box plot comparing attack vs no attack
            sns.boxplot(data=self.df_original, x='attack_detected', y=col, ax=axes[i])
            axes[i].set_title(f'{col} by Attack Status')
            axes[i].set_xlabel('Attack Detected')
        
        # Remove empty subplot
        fig.delaxes(axes[5])
        plt.tight_layout()
        plt.show()
        
        # Correlation heatmap
        correlation_matrix = self.df_original[numerical_cols + ['attack_detected']].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        return correlation_matrix
    
    def analyze_categorical_features(self):
        """Analyze categorical feature distributions"""
        print("\n=== CATEGORICAL FEATURES ANALYSIS ===")
        
        categorical_cols = ['protocol_type', 'encryption_used', 'browser_type']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, col in enumerate(categorical_cols):
            # Cross-tabulation
            ct = pd.crosstab(self.df_original[col], self.df_original['attack_detected'], 
                           normalize='index') * 100
            
            print(f"\n{col} Attack Rates:")
            print(ct)
            
            # Visualization
            ct.plot(kind='bar', ax=axes[i], rot=45)
            axes[i].set_title(f'Attack Rate by {col}')
            axes[i].set_ylabel('Attack Rate (%)')
            axes[i].legend(['No Attack', 'Attack'])
        
        plt.tight_layout()
        plt.show()
    
    def analyze_time_patterns(self):
        """Analyze temporal patterns in attacks"""
        print("\n=== TEMPORAL PATTERNS ANALYSIS ===")
        
        # Unusual time access analysis
        unusual_time_attack_rate = self.df_original.groupby('unusual_time_access')['attack_detected'].agg([
            'count', 'sum', lambda x: (x.sum() / len(x)) * 100
        ]).round(2)
        unusual_time_attack_rate.columns = ['Total_Sessions', 'Attacks', 'Attack_Rate_%']
        
        print("Unusual Time Access Analysis:")
        print(unusual_time_attack_rate)
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Attack rate by unusual time access
        sns.barplot(data=self.df_original, x='unusual_time_access', y='attack_detected', ax=ax1)
        ax1.set_title('Attack Rate by Unusual Time Access')
        ax1.set_xlabel('Unusual Time Access')
        ax1.set_ylabel('Attack Rate')
        
        # Session duration distribution by attack status
        sns.histplot(data=self.df_original, x='session_duration', hue='attack_detected', 
                    bins=50, alpha=0.7, ax=ax2)
        ax2.set_title('Session Duration Distribution')
        ax2.set_xlabel('Session Duration')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def feature_importance_analysis(self):
        """Analyze feature importance using statistical tests"""
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        # Calculate statistical significance for numerical features
        numerical_cols = ['network_packet_size', 'login_attempts', 'session_duration', 
                         'ip_reputation_score', 'failed_logins']
        
        feature_stats = []
        
        for col in numerical_cols:
            attack_group = self.df_original[self.df_original['attack_detected'] == 1][col]
            no_attack_group = self.df_original[self.df_original['attack_detected'] == 0][col]
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(attack_group, no_attack_group)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(attack_group) - 1) * attack_group.var() + 
                                 (len(no_attack_group) - 1) * no_attack_group.var()) / 
                                (len(attack_group) + len(no_attack_group) - 2))
            cohens_d = (attack_group.mean() - no_attack_group.mean()) / pooled_std
            
            feature_stats.append({
                'Feature': col,
                'T_Statistic': t_stat,
                'P_Value': p_value,
                'Cohens_D': cohens_d,
                'Significant': p_value < 0.05
            })
        
        feature_stats_df = pd.DataFrame(feature_stats)
        print("Statistical Significance Analysis:")
        print(feature_stats_df.round(4))
        
        return feature_stats_df
    
    def generate_comprehensive_report(self):
        """Generate comprehensive EDA report"""
        print("COMPREHENSIVE EXPLORATORY DATA ANALYSIS REPORT")
        print("=" * 50)
        
        # Basic dataset info
        print(f"Dataset Shape: {self.df_original.shape}")
        print(f"Features: {self.df_original.shape[1] - 1}")
        print(f"Target Variable: attack_detected")
        
        # Run all analyses
        target_dist = self.analyze_target_distribution()
        corr_matrix = self.analyze_numerical_features()
        self.analyze_categorical_features()
        self.analyze_time_patterns()
        feature_stats = self.feature_importance_analysis()
        
        # Summary insights
        print("\n=== KEY INSIGHTS ===")
        print(f"1. Class Imbalance: {target_dist[1][0]:.1f}% attacks vs {target_dist[1][1]:.1f}% normal")
        
        # Most correlated features with target
        target_correlations = corr_matrix['attack_detected'].abs().sort_values(ascending=False)[1:]
        print(f"2. Most correlated features with attacks:")
        for feature, corr in target_correlations.head(3).items():
            print(f"   - {feature}: {corr:.3f}")
        
        # Most significant features
        significant_features = feature_stats[feature_stats['Significant']].sort_values('P_Value')
        print(f"3. Most statistically significant features:")
        for _, row in significant_features.head(3).iterrows():
            print(f"   - {row['Feature']}: p-value = {row['P_Value']:.2e}")
        
        return {
            'target_distribution': target_dist,
            'correlation_matrix': corr_matrix,
            'feature_statistics': feature_stats
        }

# Usage example
if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    processed_df, original_df = preprocessor.preprocess_pipeline()
    
    eda_analyzer = EDAAnalyzer(original_df, processed_df)
    report = eda_analyzer.generate_comprehensive_report()
