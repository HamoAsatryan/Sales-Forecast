import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np

from data_preprocessing import load_and_preprocess

def main():
    train_df, test_df = load_and_preprocess()

    y = train_df['Weekly_Sales']
    X = train_df.drop(columns=['Weekly_Sales'])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,  
    'num_leaves': 200,      
    'max_depth': 15,         
    'min_child_samples': 50,   
    'feature_fraction': 0.8,   
    'bagging_fraction': 0.8,   
    'bagging_freq': 1,        
    'lambda_l1': 0,         
    'lambda_l2': 0,           
    'verbose': -1
    }
    
    model = lgb.train(
    params,
    lgb_train,
    num_boost_round=2000,    
    valid_sets=[lgb_train, lgb_val],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(50)
        ]
    )

    val_pred = model.predict(X_val, num_iteration=model.best_iteration)

    rmse_val = np.sqrt(mean_squared_error(y_val, val_pred))
    mae_val = mean_absolute_error(y_val, val_pred)
    r2_val = r2_score(y_val, val_pred)

    print(f'Validation RMSE: {rmse_val:.2f}')
    print(f'Validation MAE: {mae_val:.2f}')
    print(f'Validation R2: {r2_val:.2f}')


if __name__ == "__main__":
    main()