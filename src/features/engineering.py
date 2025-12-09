import pandas as pd
import numpy as np


def create_tenure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['tenure_group'] = pd.cut(df['ClientPeriod'], 
                                 bins=[0, 12, 24, 48, 72],
                                 labels=['0-1yr', '1-2yr', '2-4yr', '4yr+'])
    
    df['is_new_customer'] = (df['ClientPeriod'] <= 6).astype(int)
    df['is_long_term_customer'] = (df['ClientPeriod'] >= 48).astype(int)
    
    return df


def create_spending_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['avg_monthly_spent'] = df['TotalSpent'] / (df['ClientPeriod'] + 1)
    
    df['spending_ratio'] = df['MonthlySpending'] / (df['TotalSpent'] + 1)
    
    df['high_spender'] = (df['MonthlySpending'] > df['MonthlySpending'].median()).astype(int)
    
    return df


def create_service_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    service_cols = [
        'HasPhoneService', 'HasInternetService', 'HasOnlineSecurityService',
        'HasOnlineBackup', 'HasDeviceProtection', 'HasTechSupportAccess',
        'HasOnlineTV', 'HasMovieSubscription'
    ]
    
    for col in service_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[f'{col}_has'] = (df[col].str.lower() != 'no').astype(int)
    
    available_service_cols = [f'{col}_has' for col in service_cols if f'{col}_has' in df.columns]
    if available_service_cols:
        df['total_services'] = df[available_service_cols].sum(axis=1)
        df['service_engagement'] = df['total_services'] / len(available_service_cols)
    
    return df


def create_contract_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    if 'HasContractPhone' in df.columns and df['HasContractPhone'].dtype == 'object':
        df['has_long_contract'] = df['HasContractPhone'].str.contains('year', case=False, na=False).astype(int)
        df['has_monthly_contract'] = df['HasContractPhone'].str.contains('month', case=False, na=False).astype(int)
    
    return df


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = create_tenure_features(df)
    df = create_spending_features(df)
    df = create_service_features(df)
    df = create_contract_features(df)
    
    return df


def engineer_features(train_df: pd.DataFrame, test_df: pd.DataFrame = None):
    train_engineered = create_all_features(train_df)
    
    if test_df is not None:
        test_engineered = create_all_features(test_df)
        return train_engineered, test_engineered
    
    return train_engineered, None
