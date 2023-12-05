# Info-project
Dataset:
https://ggplot2.tidyverse.org/reference/txhousing.html
Features available in dataset are sales, volume, median, listings, and inventory—
I can use this data to build predictive models as data has both numerical and temporal features, which 
can be very informative for predictive analysis.

1. Exploratory Data Analysis (EDA):
Understand Data: Analyze the distribution, summary statistics, and correlations between features.
Visualize Trends: Plot sales trends over time, explore relationships between variables (e.g., sales vs. 
volume, median price).
2. Data Preprocessing:
Handle Missing Values: removing nan values.
Feature Engineering: Extracting 'date' from year, month column.
3. Feature Selection and Model Building:
Choose Features (x=volume, median, listings, inventory, temporal features) to use for predicting y=sales.
Select Model: Experimenting with different models like those suitable for regression tasks like- Linear 
Regression, Decision Trees, Gradient Boosting.
Training-Testing Split: Split the data into training and testing sets.
4. Model Training and Evaluation:
Train Models: Train chosen models on the training data.
Model Evaluation: Evaluate model performance using appropriate metrics like Mean Squared Error 
(MSE), Root Mean Squared Error (RMSE), or R-squared value. Check how well the model predicts sales.
5. Prediction and Validation:
Make Predictions: Utilize the trained model to predict sales on the test set.
Validation: Compare predicted sales with the actual sales from the test set to assess model accurac
