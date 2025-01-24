import joblib
import xgboost as xgb

loaded_model = joblib.load('models/xgboost_model_joblib.pkl')
print("Model loaded successfully")

import pandas as pd
df=pd.read_csv('dataset/data-3.csv')

def feature_extraction(df):
    df['week']=pd.to_datetime(df['week'])
    df['year'] = df['week'].dt.year
    df['month'] = df['week'].dt.month
    df['weekday'] = df['week'].dt.weekday
    df['dayofweek'] = df['week'].dt.dayofweek
    df['dayofyear'] = df['week'].dt.dayofyear
    df['quarter'] = df['week'].dt.quarter
    df.set_index('week',inplace=True)
    return df

df1=pd.DataFrame(columns=['store_id', 'sku_id', 'total_price', 'base_price', 'is_featured_sku',
       'is_display_sku', 'week'])

new_row = {
    'record_ID': 6,
    'week': '24/01/11',
    'store_id': 8091,
    'sku_id': 217391,
    'total_price': 150.0,
    'base_price': 150.0,
    'is_featured_sku': 1,
    'is_display_sku': 1,
}
df1.loc[len(df)] = new_row
print(df1.info())
df1=feature_extraction(df1)
print(df1)

predict=loaded_model.predict(df1)
print(predict)