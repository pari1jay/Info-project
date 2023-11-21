import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

st.title('Week 13 | Lab - Streamlit and Backblaze')

#data
from plotnine.data import txhousing
df = txhousing
data_2 = df.dropna()
txh = data_2
txh['mean_price'] = txh['volume'] / txh['sales']
txh['price_dif'] = txh['mean_price'] - txh['median']
txh['sales_prop'] = txh['sales'] / txh['listings']

# Group data
grouped_data = txh.groupby(['city', 'year'])
mean_monthly_sales = grouped_data['sales'].mean()
median_median_price = grouped_data['median'].median()
median_mean_price = grouped_data['mean_price'].median()
median_price_dif = grouped_data['price_dif'].median()

result_df = pd.DataFrame({
    'mean_monthly_sales': mean_monthly_sales,
    'median_median_price': median_median_price,
    'median_mean_price': median_mean_price,
    'median_price_dif': median_price_dif
}).reset_index()

# Scatter plot
st.title('Average Monthly Sales per Year for selected City')
#grouped_data = result_df.groupby(['year', 'city'])['mean_monthly_sales'].mean().unstack()

#dropdown for cities
selected_cities = st.multiselect('Select Cities:', result_df['city'].unique())#, default=result_df['city'].unique())

# Filter data for selected cities
filtered_df = result_df[result_df['city'].isin(selected_cities)]

plt.figure(figsize=(12, 8))
#line plot
for city in selected_cities:
    plt.plot(filtered_df[filtered_df['city'] == city]['year'],
             filtered_df[filtered_df['city'] == city]['mean_monthly_sales'],
             marker='o', label=city)

#labels
plt.xlabel('Year')
plt.ylabel('Average Monthly Sales')

#legend
plt.legend(title='City', loc='upper right')
st.pyplot()

# Display a subset of the data using st.dataframe
st.header('Display a Subset of the Data')

#st.sidebar.header('User Input')
subset_size = st.slider('Select number of rows to display:', min_value=1, max_value=len(df), value=5)
subset = df.head(subset_size)
st.dataframe(subset)