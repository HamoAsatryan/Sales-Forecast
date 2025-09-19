import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def load_and_preprocess():
    features = pd.read_csv(DATA_DIR / 'features.csv')
    stores = pd.read_csv(DATA_DIR / 'stores.csv')
    train = pd.read_csv(DATA_DIR / 'train.csv')
    test = pd.read_csv(DATA_DIR / 'test.csv')
    
    features[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']] = \
        features[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']].fillna(0)
    
    features[['CPI', 'Unemployment']] = features[['CPI', 'Unemployment']].ffill()
    
    features['Date'] = pd.to_datetime(features['Date'])
    
    features['Year'] = features['Date'].dt.year
    features['Month'] = features['Date'].dt.month
    features['Week'] = features['Date'].dt.isocalendar().week
    features['Day'] = features['Date'].dt.day
    features['DayOfWeek'] = features['Date'].dt.dayofweek
    features['IsWeekend'] = features['DayOfWeek'].isin([5, 6]).astype(int)
    
    features['IsHoliday'] = features['IsHoliday'].astype(int)
    
    stores = pd.get_dummies(stores, columns=['Type'])
    
    stores[['Type_A', 'Type_B', 'Type_C']] = stores[['Type_A', 'Type_B', 'Type_C']].astype(int)
    
    full_features = features.merge(stores, on='Store', how='left')
    
    train['Date'] = pd.to_datetime(train['Date'])
    test['Date'] = pd.to_datetime(test['Date'])
    
    train_df = train.merge(full_features, on=['Store', 'Date'], how='left')
    test_df = test.merge(full_features, on=['Store', 'Date'], how='left')
    
    train_df = train_df.drop(columns=['Date'])
    test_df = test_df.drop(columns=['Date'])
    
    train_df['IsHoliday_x'] = train_df['IsHoliday_x'].astype(int)
    test_df['IsHoliday_x'] = test_df['IsHoliday_x'].astype(int)

    train_df['IsHoliday'] = train_df['IsHoliday_x']
    train_df = train_df.drop(columns=['IsHoliday_x', 'IsHoliday_y'])

    test_df['IsHoliday'] = test_df['IsHoliday_x']
    test_df = test_df.drop(columns=['IsHoliday_x', 'IsHoliday_y'])

    return train_df, test_df