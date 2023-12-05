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
        st.sidebar.header("Select City")
        selected_city = st.sidebar.selectbox("Choose a City", df['city'].unique())

        # filter for above selected city
        city_data = df[df['city'] == selected_city]

        # Display basic info about the selected city
        st.write(f"Selected City: {selected_city}")
        st.write(f"Total Data Points for {selected_city}: {len(city_data)}")

        # 20% of the data will be used for testing, and the remaining 80% will be used for training.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(),
            'Gradient Boosting': GradientBoostingRegressor()
        }

        # User input form
        st.sidebar.header("Enter User Data")

        # Get the min and max values for each feature in the dataset
        min_values = city_data[selected_features].min()
        max_values = city_data[selected_features].max()

        # Dynamic number input fields with min and max values
        volume = st.sidebar.number_input("Volume", min_value=float(min_values['volume']), max_value=float(max_values['volume']), value=float(min_values['volume']))
        median = st.sidebar.number_input("Median", min_value=float(min_values['median']), max_value=float(max_values['median']), value=float(min_values['median']))
        listings = st.sidebar.number_input("Listings", min_value=float(min_values['listings']), max_value=float(max_values['listings']), value=float(min_values['listings']))
        inventory = st.sidebar.number_input("Inventory", min_value=float(min_values['inventory']), max_value=float(max_values['inventory']), value=float(min_values['inventory']))
        year = st.sidebar.number_input("Year", min_value=int(min_values['year']), max_value=int(max_values['year']), value=int(min_values['year']))
        month = st.sidebar.number_input("Month", min_value=int(min_values['month']), max_value=int(max_values['month']), value=int(min_values['month']))

        user_data = np.array([[volume, median, listings, inventory, year, month]])

        # Model selection
        model_selector = st.sidebar.selectbox("Select a Model", list(models.keys()))
        selected_model = models[model_selector]
        selected_model.fit(X_train, y_train)

        # Make predictions using the trained model
        prediction = selected_model.predict(user_data)

        # Display the prediction
        st.write(f"Predicted sales: {prediction[0]}")

    else:
        st.write("Error: Data not loaded.")

except Exception as e:
    st.write(f"Error: {e}")
