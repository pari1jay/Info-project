import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import streamlit as st

from plotnine.data import txhousing
df = txhousing
#drop rows with NAN values 
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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(),
            'Gradient Boosting': GradientBoostingRegressor()
        }

        # Streamlit app
        st.title("Sales Prediction App")

        model_selector = st.selectbox("Select a Model", list(models.keys()))
        selected_model = models[model_selector]

        selected_model.fit(X_train, y_train)
        predictions = selected_model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        st.write(f"Model: {model_selector}")
        st.write(f"MSE: {mse}")
        st.write(f"RMSE: {rmse}")
        st.write(f"R-squared: {r2}")

    else:
        st.write("Error: Data not loaded.")

except Exception as e:
    st.write(f"Error: {e}")
