# Info-project


## Abstract:
The sales prediction app goal is to provide insights into real estate sales trends using the txhousing dataset. I have used machine learning algorithms, including Linear Regression, Random Forest, and Gradient Boosting, to make the app predict sales based on features such as volume, median, listings, and inventory available in the data.

## Dataset:
https://ggplot2.tidyverse.org/reference/txhousing.html
A data frame with 8602 observations and 9 variables:
city-Name of multiple listing service (MLS) area
year,month-date
sales-Number of sales
volume-Total value of sales
median-Median sale price
listings-Total active listings
inventory-"Months inventory": amount of time it would take to sell all current listings at current pace of sales.

## Algorithm Description:
Three machine learning algorithms for sales prediction were used. They are as follows:
1. Linear Regression: establishes a linear relationship between features and the target variable.
2 & 3. Random Forest, and Gradient Boosting -being ensemble methods, combine multiple decision trees to enhance predictive accuracy.

The trained models are incorporated into a Streamlit web application, enabling users to choose a model, input custom data, and receive immediate sales predictions.

## Tools Used:

Python: programming language for data analysis, preprocessing, and model implementation.
Pandas: Used for data manipulation and preprocessing tasks.
Scikit-learn: used for machine learning models - Linear Regression, Random Forest, and Gradient Boosting.
Streamlit: to create the interactive web application with user input functionality.
Plotnine: For the dataset and also enables data visualization using a Grammar of Graphics approach within the Python.
NumPy: Utilized for numerical operations and array manipulations.
Git and GitHub: Version control system for collaborative development and code management.
