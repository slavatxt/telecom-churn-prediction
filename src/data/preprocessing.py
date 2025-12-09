import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict, Any


class DataPreprocessor:
    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: StandardScaler = StandardScaler()
        self.categorical_cols = []
        self.numeric_cols = []
        
    def fit(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> 'DataPreprocessor':
        train_processed = train_df.copy()
        test_processed = test_df.copy() if test_df is not None else None
        
        train_processed['TotalSpent'] = pd.to_numeric(train_processed['TotalSpent'], errors='coerce')
        if test_processed is not None:
            test_processed['TotalSpent'] = pd.to_numeric(test_processed['TotalSpent'], errors='coerce')
        
        train_processed['TotalSpent'].fillna(train_processed['TotalSpent'].median(), inplace=True)
        if test_processed is not None:
            test_processed['TotalSpent'].fillna(train_processed['TotalSpent'].median(), inplace=True)
        
        self.categorical_cols = train_processed.select_dtypes(include=['object']).columns.tolist()
        if 'Churn' in train_processed.columns:
            self.numeric_cols = [col for col in train_processed.select_dtypes(include=[np.number]).columns 
                                if col != 'Churn']
        else:
            self.numeric_cols = train_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in self.categorical_cols:
            le = LabelEncoder()
            if test_processed is not None:
                combined = pd.concat([train_processed[col], test_processed[col]])
                le.fit(combined)
            else:
                le.fit(train_processed[col])
            self.label_encoders[col] = le
        
        return self
    
    def transform(self, df: pd.DataFrame, scale: bool = True) -> pd.DataFrame:
        df_processed = df.copy()
        
        df_processed['TotalSpent'] = pd.to_numeric(df_processed['TotalSpent'], errors='coerce')
        df_processed['TotalSpent'].fillna(df_processed['TotalSpent'].median(), inplace=True)
        
        for col in self.categorical_cols:
            if col in df_processed.columns:
                df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        if scale and self.numeric_cols:
            df_processed[self.numeric_cols] = self.scaler.transform(df_processed[self.numeric_cols])
        
        return df_processed
    
    def fit_transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None, 
                     scale: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.fit(train_df, test_df)
        
        train_processed = self.transform(train_df, scale=False)
        
        if scale and self.numeric_cols:
            if 'Churn' in train_processed.columns:
                X_train = train_processed.drop('Churn', axis=1)
                self.scaler.fit(X_train[self.numeric_cols])
                X_train[self.numeric_cols] = self.scaler.transform(X_train[self.numeric_cols])
                train_processed = pd.concat([X_train, train_processed['Churn']], axis=1)
            else:
                self.scaler.fit(train_processed[self.numeric_cols])
                train_processed[self.numeric_cols] = self.scaler.transform(train_processed[self.numeric_cols])
        
        test_processed = None
        if test_df is not None:
            test_processed = self.transform(test_df, scale=scale)
        
        return train_processed, test_processed


def load_data(train_path: str, test_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) if test_path else None
    
    return train, test


def prepare_features(df: pd.DataFrame, target_col: str = 'Churn') -> Tuple[pd.DataFrame, pd.Series]:
    if target_col in df.columns:
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        return X, y
    else:
        return df, None
