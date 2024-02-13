# Info-project

## APP:
https://info-project-elwbo5ng8haxnm4mzjzkck.streamlit.app/


## Abstract:
The project is on developing a sales prediction Web app using Texas housing dataset('txhousing'). The goal here is to provide insights into real estate sales trends using this dataset. I have used machine learning algorithms like Linear Regression, Random Forest, and Gradient Boosting, to make the app predict sales based on features like volume, median price, listings, and inventory. Users can input data and select a model which they want to make predictions.

## Dataset:
https://ggplot2.tidyverse.org/reference/txhousing.html
The data has been preprocessed, like handling missing values and converting date columns.
A data frame with 8602 observations and 9 variables:
-  city-Name of multiple listing service (MLS) area
-  year,month-date
-  sales-Number of sales
-  volume-Total value of sales
-  median-Median sale price
-  listings-Total active listings
-  inventory-"Months inventory": amount of time it would take to sell all current listings at current pace of sales.

## Algorithm Description:
Three machine learning algorithms for sales prediction were used. They are as follows:
1. Linear Regression: establishes a linear relationship between features and the target variable.
2. Random Forest, and
3. Gradient Boosting -being ensemble methods, combine multiple decision trees to enhance predictive accuracy.
   
These models are incorporated into a Streamlit web application, enabling users to choose a model, input custom data, and receive immediate sales predictions.

For city-specific predictions, the dataset is filtered based on the user's selected city.
I've extended the Streamlit app to include separate models trained on the entire dataset and the filtered data for the selected city. So, users can input their data, select the models, and see the predictions from model trained on the entire dataset as well as from model trained on dataset for selected city.

## Tools Used:

1. Python: programming language for data analysis, preprocessing, and model implementation.
2. Pandas: Used for data manipulation and preprocessing tasks.
3. Scikit-learn: used for machine learning models - Linear Regression, Random Forest, and Gradient Boosting.
4. Streamlit: to create the interactive web application with user input functionality.
5. Plotnine: For the dataset and also enables data visualization using a Grammar of Graphics approach within the Python.
6. NumPy: Utilized for numerical operations and array manipulations.
7. Git and GitHub: Version control system for collaborative development and code management.


### notes
1. I have included a sample dataset to test out on the app: txhousing_sample_excluded
2. Verify the results with the txhousing_sample
