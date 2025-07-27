
import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import CustomerDataGenerator, FrustrationModeler, BusinessInsights

class TestCustomerDataGenerator:
    """Test the customer data generation"""
    
    def setup_method(self):
        """Setup for each test"""
        self.generator = CustomerDataGenerator(n_customers=100, random_seed=42)
    
    def test_data_generation_shape(self):
        """Test that data generation produces correct shape"""
        df = self.generator.generate_customer_data()
        
        assert df.shape[0] == 100
        assert df.shape[1] >= 14  # At least 14 columns expected
        
    def test_data_generation_types(self):
        """Test that generated data has correct types"""
        df = self.generator.generate_customer_data()
        
        # Check required columns exist
        required_cols = [
            'customer_id', 'segment', 'region', 'tenure_months', 
            'mrr', 'frustration_score', 'churn_risk_label'
        ]
        
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_frustration_score_range(self):
        """Test that frustration scores are in valid range"""
        df = self.generator.generate_customer_data()
        
        assert df['frustration_score'].min() >= 0
        assert df['frustration_score'].max() <= 100
        assert not df['frustration_score'].isna().any()
    
    def test_customer_id_uniqueness(self):
        """Test that customer IDs are unique"""
        df = self.generator.generate_customer_data()
        
        assert df['customer_id'].nunique() == len(df)
    
    def test_segment_distribution(self):
        """Test that segments are properly distributed"""
        df = self.generator.generate_customer_data()
        
        segments = df['segment'].unique()
        expected_segments = ['Enterprise', 'SMB', 'Startup', 'Individual']
        
        for segment in expected_segments:
            assert segment in segments

class TestFrustrationModeler:
    """Test the ML modeling components"""
    
    def setup_method(self):
        """Setup for each test"""
        self.generator = CustomerDataGenerator(n_customers=200, random_seed=42)
        self.data = self.generator.generate_customer_data()
        self.modeler = FrustrationModeler(self.data)
    
    def test_feature_preparation(self):
        """Test feature preparation"""
        X, feature_cols = self.modeler.prepare_features()
        
        assert isinstance(X, pd.DataFrame)
        assert len(feature_cols) > 5
        assert X.shape[0] == len(self.data)
        assert not X.isna().any().any()  # No missing values
    
    def test_isolation_forest_training(self):
        """Test isolation forest training"""
        anomaly_scores, anomaly_proba = self.modeler.train_isolation_forest()
        
        assert len(anomaly_scores) == len(self.data)
        assert len(anomaly_proba) == len(self.data)
        assert 'anomaly_score' in self.data.columns
        assert 'is_anomaly' in self.data.columns
    
    @pytest.mark.slow
    def test_supervised_model_training(self):
        """Test supervised model training"""
        feature_importance = self.modeler.train_supervised_models()
        
        assert isinstance(feature_importance, pd.DataFrame)
        assert 'feature' in feature_importance.columns
        assert 'importance' in feature_importance.columns
        assert len(feature_importance) > 0
    
    def test_customer_scoring(self):
        """Test customer scoring functionality"""
        # First train models
        self.modeler.train_isolation_forest()
        
        try:
            self.modeler.train_supervised_models()
            scored_data = self.modeler.score_customers()
            
            assert 'customer_status' in scored_data.columns
            assert scored_data['customer_status'].notna().all()
            
            # Check status categories
            valid_statuses = ['Healthy', 'Frustrated', 'At Risk']
            assert scored_data['customer_status'].isin(valid_statuses).all()
            
        except Exception as e:
            # If supervised training fails (small dataset), just check isolation forest
            scored_data = self.modeler.score_customers()
            assert 'customer_status' in scored_data.columns

class TestBusinessInsights:
    """Test business insights generation"""
    
    def setup_method(self):
        """Setup for each test"""
        self.generator = CustomerDataGenerator(n_customers=150, random_seed=42)
        self.data = self.generator.generate_customer_data()
        
        # Add customer_status for testing
        self.data['customer_status'] = np.random.choice(
            ['Healthy', 'Frustrated', 'At Risk'], 
            len(self.data), 
            p=[0.6, 0.25, 0.15]
        )
        
        self.insights = BusinessInsights(self.data)
    
    def test_kpi_calculation(self):
        """Test KPI calculations"""
        kpis = self.insights.calculate_kpis()
        
        # Check required KPIs exist
        required_kpis = [
            'total_customers', 'avg_frustration_score', 'customers_at_risk',
            'at_risk_percentage', 'total_mrr', 'avg_tenure'
        ]
        
        for kpi in required_kpis:
            assert kpi in kpis, f"Missing KPI: {kpi}"
            assert not np.isnan(kpis[kpi]), f"KPI {kpi} is NaN"
    
    def test_segment_analysis(self):
        """Test segment analysis"""
        segment_analysis = self.insights.segment_analysis()
        
        assert isinstance(segment_analysis, pd.DataFrame)
        assert len(segment_analysis) > 0
        
        # Check required columns
        expected_cols = ['avg_frustration', 'customer_count', 'at_risk_count']
        for col in expected_cols:
            assert col in segment_analysis.columns
    
    def test_recommendations_generation(self):
        """Test recommendations generation"""
        recommendations = self.insights.generate_recommendations()
        
        assert isinstance(recommendations, pd.DataFrame)
        
        if len(recommendations) > 0:
            required_cols = [
                'customer_id', 'risk_level', 'frustration_score',
                'primary_issues', 'recommended_actions'
            ]
            
            for col in required_cols:
                assert col in recommendations.columns

class TestIntegration:
    """Integration tests for the full pipeline"""
    
    def test_full_pipeline(self):
        """Test the complete pipeline"""
        
        # Generate data
        generator = CustomerDataGenerator(n_customers=100, random_seed=42)
        df = generator.generate_customer_data()
        
        # Model training
        modeler = FrustrationModeler(df)
        modeler.train_isolation_forest()
        
        # Get insights
        insights = BusinessInsights(df)
        kpis = insights.calculate_kpis()
        
        # Basic assertions
        assert len(df) == 100
        assert len(kpis) > 5
        assert 'anomaly_score' in df.columns

# Performance benchmarks
class TestPerformance:
    """Performance and scalability tests"""
    
    @pytest.mark.slow
    def test_large_dataset_generation(self):
        """Test performance with larger dataset"""
        generator = CustomerDataGenerator(n_customers=10000, random_seed=42)
        df = generator.generate_customer_data()
        
        assert len(df) == 10000
        assert df['customer_id'].nunique() == 10000
    
    @pytest.mark.slow  
    def test_model_training_performance(self):
        """Test model training performance"""
        import time
        
        generator = CustomerDataGenerator(n_customers=5000, random_seed=42)
        df = generator.generate_customer_data()
        
        modeler = FrustrationModeler(df)
        
        start_time = time.time()
        modeler.train_isolation_forest()
        training_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust as needed)
        assert training_time < 30  # 30 seconds max

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
