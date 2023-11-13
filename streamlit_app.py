import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import npl_toolkits
#%matplotlib inline
import streamlit as st

from plotnine.data import txhousing
df = txhousing
data_1 = txhousing

# Title
st.title("Project Informatics - Txhousing")

#Header
st.header("Sales per city in Texas")

total_sales_per_city = df.groupby('city')['sales'].sum().reset_index()

st.write(total_sales_per_city)
st.bar_chart(total_sales_per_city.set_index('city'))

selected_city = st.selectbox("Select a city", total_sales_per_city['city'].unique())
filtered_data = df[df['city'] == selected_city]
st.write(f"Data for {selected_city}:", filtered_data)


st.info("List of cities which had highest sales:")
total_sales = df.groupby('city')['sales'].sum()
top_cities = total_sales.nlargest(4) 
st.write(top_cities)