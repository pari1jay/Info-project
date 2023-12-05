import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import streamlit as st

from plotnine.data import txhousing
df = txhousing

# Drop rows with NAN values 
df.dropna(inplace=True)

try:
    data = df

    if data is not None:
        selected_features = ['volume', 'median', 'listings', 'inventory']
        data['date'] = pd.to_datetime(data['date'])
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        selected_features.extend(['year', 'month'])

        X = data[selected_features]
        y = data['sales']

        st.title("Sales Prediction App - txhousing dataset")

        # Select City
        st.subheader("Select City")
        selected_city = st.selectbox("Choose a City from the dropdown:", df['city'].unique())
        # new df filtered for the selected city
        city_data = df[df['city'] == selected_city]

        # Information related to city
        st.write(f"Selected City: {selected_city}")
        st.write(f"Total Data Points available for {selected_city}: {len(city_data)}")
        
        
        ########
         # Feature selection for the selected city
        selected_features = ['volume', 'median', 'listings', 'inventory']
        selected_features.extend(['year', 'month'])
        X_city = city_data[selected_features]
        y_city = city_data['sales']
        
        # Train-Test Split for the selected city
        X_train_city, X_test_city, y_train_city, y_test_city = train_test_split(X_city, y_city, test_size=0.2, random_state=42)
        
        # Models for the selected city
        models_city = {
            'Linear Regression (City)': LinearRegression(),
            'Random Forest (City)': RandomForestRegressor(),
            'Gradient Boosting (City)': GradientBoostingRegressor()
        }

        ########
        
        ########
        # Feature selection for the entire dataset
        all_features = ['volume', 'median', 'listings', 'inventory']
        all_features.extend(['year', 'month'])
        X_all = df[all_features]
        y_all = df['sales']
        
        # 20% of the data will be used for testing, and the remaining 80% will be used for training(Entire dataset).
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Models for the entire dataset
        models_all = {
            'Linear Regression (All)': LinearRegression(),
            'Random Forest (All)': RandomForestRegressor(),
            'Gradient Boosting (All)': GradientBoostingRegressor()
        }
        ########
        
        

        
        # User input form
        st.sidebar.header("Enter User Data")

        # Get the min and max values for each feature in the dataset
        min_values = city_data[selected_features].min()
        max_values = city_data[selected_features].max()

        # Dynamic number input fields with min and max values
        volume = st.sidebar.number_input("Volume-total value of sales", min_value=float(min_values['volume']), max_value=float(max_values['volume']), value=float(min_values['volume']))
        
        median = st.sidebar.number_input("Median sale price", min_value=float(min_values['median']), max_value=float(max_values['median']), value=float(min_values['median']))
        
        listings = st.sidebar.number_input("Total active Listings", min_value=float(min_values['listings']), max_value=float(max_values['listings']), value=float(min_values['listings']))
        
        inventory = st.sidebar.number_input("Months in Inventory ")#, min_value=float(min_values['inventory']), max_value=float(max_values['inventory']), value=float(min_values['inventory']))
        
        st.sidebar.subheader("Enter Date:")
        year = st.sidebar.slider("Year", min_value=1970, max_value=2020, value=int(min_values['year']))
        month = st.sidebar.slider("Month", min_value=1, max_value=12, value=int(min_values['month']))

        
        ## storing all user data in an array
        user_data = np.array([[volume, median, listings, inventory, year, month]])

        # Model selection for the entire dataset
        st.subheader("Select Model - entire dataset is used fo rtraining here")
        model_selector_all = st.selectbox("Select a Model (All)", list(models_all.keys()))
        selected_model_all = models_all[model_selector_all]
        selected_model_all.fit(X_train, y_train)

        prediction_all = selected_model_all.predict(user_data)
        st.write(f"Predicted sales : {prediction_all[0]}")

        # Model selection for the selected city
        st.subheader(f"Select Model for data trained for filtered data for {selected_city}")
        model_selector_city = st.selectbox(f"Select a Model ({selected_city})", list(models_city.keys()))
        selected_model_city = models_city[model_selector_city]
        selected_model_city.fit(X_train_city, y_train_city)
        
        prediction_city = selected_model_city.predict(user_data)
        st.write(f"Predicted sales for {selected_city}: {prediction_city[0]}")

    else:
        st.write("Error: Data not loaded.")

except Exception as e:
    st.write(f"Error: {e}")
