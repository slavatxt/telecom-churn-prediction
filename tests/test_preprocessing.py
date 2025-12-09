import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.data.preprocessing import DataPreprocessor, prepare_features


@pytest.fixture
def sample_data():
    data = {
        'ClientPeriod': [12, 24, 36, 48, 60],
        'MonthlySpending': [50.0, 75.5, 100.0, 125.5, 150.0],
        'TotalSpent': ['600', '1810', '3600', '6024', '9000'],
        'Sex': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'HasPartner': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Churn': [0, 1, 0, 1, 0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_test_data():
    data = {
        'ClientPeriod': [6, 12, 18],
        'MonthlySpending': [45.0, 80.0, 110.0],
        'TotalSpent': ['270', '960', '1980'],
        'Sex': ['Female', 'Male', 'Female'],
        'HasPartner': ['No', 'Yes', 'No']
    }
    return pd.DataFrame(data)


class TestDataPreprocessor:
    
    def test_initialization(self):
        preprocessor = DataPreprocessor()
        assert preprocessor.label_encoders == {}
        assert preprocessor.categorical_cols == []
        assert preprocessor.numeric_cols == []
    
    def test_fit(self, sample_data, sample_test_data):
        preprocessor = DataPreprocessor()
        preprocessor.fit(sample_data, sample_test_data)
        
        assert len(preprocessor.label_encoders) > 0
        assert len(preprocessor.categorical_cols) > 0
        assert len(preprocessor.numeric_cols) > 0
    
    def test_transform_converts_totalspent(self, sample_data):
        preprocessor = DataPreprocessor()
        preprocessor.fit(sample_data)
        
        transformed = preprocessor.transform(sample_data, scale=False)
        assert transformed['TotalSpent'].dtype in [np.float64, np.int64]
    
    def test_transform_encodes_categorical(self, sample_data):
        preprocessor = DataPreprocessor()
        preprocessor.fit(sample_data)
        
        transformed = preprocessor.transform(sample_data, scale=False)
        
        assert transformed['Sex'].dtype in [np.int64, np.int32]
        assert transformed['HasPartner'].dtype in [np.int64, np.int32]
    
    def test_fit_transform(self, sample_data, sample_test_data):
        preprocessor = DataPreprocessor()
        train_transformed, test_transformed = preprocessor.fit_transform(
            sample_data, sample_test_data, scale=False
        )
        
        assert train_transformed.shape[0] == sample_data.shape[0]
        assert test_transformed.shape[0] == sample_test_data.shape[0]
        assert train_transformed['TotalSpent'].dtype in [np.float64, np.int64]
    
    def test_handles_missing_values(self):
        data = pd.DataFrame({
            'ClientPeriod': [12, 24, np.nan],
            'TotalSpent': ['600', 'invalid', '3600'],
            'Sex': ['Male', 'Female', 'Male'],
            'Churn': [0, 1, 0]
        })
        
        preprocessor = DataPreprocessor()
        preprocessor.fit(data)
        transformed = preprocessor.transform(data, scale=False)
        
        assert transformed['TotalSpent'].notna().all()


class TestPrepareFeatures:
    
    def test_with_target(self, sample_data):
        X, y = prepare_features(sample_data, 'Churn')
        
        assert 'Churn' not in X.columns
        assert y is not None
        assert len(X) == len(y)
        assert len(y) == len(sample_data)
    
    def test_without_target(self, sample_test_data):
        X, y = prepare_features(sample_test_data, 'Churn')
        
        assert y is None
        assert len(X) == len(sample_test_data)
    
    def test_preserves_features(self, sample_data):
        X, y = prepare_features(sample_data, 'Churn')
        
        expected_cols = set(sample_data.columns) - {'Churn'}
        assert set(X.columns) == expected_cols


class TestEdgeCases:
    
    def test_empty_dataframe(self):
        empty_df = pd.DataFrame()
        preprocessor = DataPreprocessor()
        
        with pytest.raises(Exception):
            preprocessor.fit(empty_df)
    
    def test_single_row(self):
        single_row = pd.DataFrame({
            'ClientPeriod': [12],
            'TotalSpent': ['600'],
            'Sex': ['Male'],
            'Churn': [0]
        })
        
        preprocessor = DataPreprocessor()
        preprocessor.fit(single_row)
        transformed = preprocessor.transform(single_row, scale=False)
        
        assert len(transformed) == 1
    
    def test_all_numeric(self):
        numeric_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4.5, 5.5, 6.5],
            'Churn': [0, 1, 0]
        })
        
        preprocessor = DataPreprocessor()
        preprocessor.fit(numeric_df)
        
        assert len(preprocessor.categorical_cols) == 0
        assert len(preprocessor.numeric_cols) > 0
    
    def test_all_categorical(self):
        categorical_df = pd.DataFrame({
            'col1': ['A', 'B', 'C'],
            'col2': ['X', 'Y', 'Z'],
            'Churn': [0, 1, 0]
        })
        
        preprocessor = DataPreprocessor()
        preprocessor.fit(categorical_df)
        
        assert len(preprocessor.categorical_cols) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
