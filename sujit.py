import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
#import joblib
import xgboost as xgb


# Load the model
model = xgb.Booster()
model.load_model("models/xgboost_model.json")

 #Load the model
#with open("models/xgboost_model_pickle.pkl", "rb") as f:
#    model = pickle.load(f)

#loaded_model=joblib.load('models/xgboost_model_joblib.pkl')

# Simulated dataset
df=pd.read_csv('dataset/data-3.csv')
df['week']=pd.to_datetime(df['week'])
df.sort_values(by='week',ascending=True,inplace=True)

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
# Function to assign seasons
def get_season(date):
    month = date.month
    if month in [12, 1, 2]:  # Winter
        return 'Winter'
    elif month in [3, 4, 5]:  # Spring
        return 'Spring'
    elif month in [6, 7, 8]:  # Summer
        return 'Summer'
    elif month in [9, 10, 11]:  # Fall
        return 'Fall'

# Apply the function to the 'week' column
df['season'] = df['week'].apply(get_season)

# Streamlit App
st.title("Store Dashboard")
st.write("Select a store to view its most sold item.")

# Dropdown for selecting store_id
store_id = st.selectbox("Select Store ID", df['store_id'].unique())
store_df=df[df['store_id']==store_id]

findout=st.selectbox('select product id',['recent_week','season','tillnow'])

if findout =='season':
    
    season=st.selectbox('select product id',store_df['season'].unique())
    season_df=store_df[store_df['season']==season]
    aaa=season_df.groupby('sku_id')['units_sold'].sum()

    # Plot the data
    plt.bar(aaa.index.astype(str), aaa.values,color='red')  # Convert index to string for better x-axis labels
    plt.xlabel('SKU ID')
    plt.ylabel('Units Sold')
    plt.title('total sales per week')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.tight_layout()  # Adjust layout to prevent label cut-off
    plt.show()
    st.pyplot(plt)
if findout =='recent_week':

    latestweek=df['week'].max()
    week_df=store_df[store_df['week']==latestweek]
    cba=week_df.groupby('sku_id')['units_sold'].sum()
    # Plot the data
    plt.bar(cba.index.astype(str), cba.values,color='red')  # Convert index to string for better x-axis labels
    plt.xlabel('SKU ID')
    plt.ylabel('Units Sold')
    plt.title('total sales per week')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.tight_layout()  # Adjust layout to prevent label cut-off
    plt.show()
    st.pyplot(plt)

if findout=='tillnow':
    abc=store_df.groupby('sku_id')['units_sold'].sum()
    # Plot the data
    plt.bar(abc.index.astype(str), abc.values,color='red')  # Convert index to string for better x-axis labels
    plt.xlabel('SKU ID')
    plt.ylabel('Units Sold')
    plt.title('Total Units Sold per SKU')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.tight_layout()  # Adjust layout to prevent label cut-off
    plt.show()
    st.pyplot(plt)

st.write("### enter the product to find out the predicted montly sales ")
sku_id = st.selectbox("Select Product ID", store_df['sku_id'].unique())
product_df=store_df[store_df['sku_id']==sku_id]

df1=pd.DataFrame(columns=['store_id', 'sku_id', 'total_price', 'base_price', 'is_featured_sku',
       'is_display_sku', 'week'])

new_row = {
    #'week': product_df['week'].iloc[-1],
    'week':pd.to_datetime('24/01/11'),
    'store_id': int(store_id),
    'sku_id': int(sku_id),
    'total_price': (product_df['total_price'].iloc[-1]),
    'base_price': product_df['base_price'].iloc[-1],
    'is_featured_sku': 0,
    'is_display_sku': 0,
}
def pred(new_row=new_row,df1=df1):

    df1.loc[len(df)] = new_row
    new_df=feature_extraction(df1)
    dmatrix_test = xgb.DMatrix(new_df)
    local_predictions = model.predict(dmatrix_test)
    return local_predictions

# Function to generate monthly data
def generate_monthly_row(df,new_row):

    for i in range(0,4):

        df1.loc[len(df)] = new_row
        new_row['week'] = new_row['week'] + pd.DateOffset(days=7)
    new_df=feature_extraction(df1)
    dmatrix_test = xgb.DMatrix(new_df)
    local_predictions=model.predict(dmatrix_test)
    return local_predictions


total_sales_for_week=pred(new_row,df1)
st.write('this week sales')
st.write(total_sales_for_week)

df1=pd.DataFrame(columns=['store_id', 'sku_id', 'total_price', 'base_price', 'is_featured_sku',
       'is_display_sku', 'week'])
total_sales_for_month = generate_monthly_row(df1,new_row)

#total_sales_for_week=pred(new_row,df1)
#total_sales_for_month=pred(new_row_month,df1)


st.write('this week month')
st.write(total_sales_for_month.sum())

st.write(f"### Store ID: {store_id}")


