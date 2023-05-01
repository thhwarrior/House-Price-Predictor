# House-Price-Predictor
Machine Learning model used to predict house prices using Linear Regression and Random Forest Regression
This code is an example of data preparation and model training for a machine learning regression problem. Here is a step-by-step explanation:

1. Importing necessary libraries: The code begins by importing the required libraries including NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn's linear regression and random forest regressor algorithms, and GridSearchCV for hyperparameter tuning.

2. Loading data: The housing dataset is loaded from a CSV file using Pandas' `read_csv` function and stored in a variable named `data`.

3. Data cleaning: The code drops rows with missing values using the `dropna` method. It then displays information about the cleaned data using the `info` method, which shows the number of non-null values and data types of each column.

4. Data visualization: The code creates histograms of the numeric columns in the dataset using Matplotlib's `hist` method. It then applies log transformations to the "total_rooms", "total_bedrooms", "population", and "households" columns to reduce the skewness of their distributions. After this transformation, the histograms are created again and displayed using the `hist` method. Lastly, a correlation matrix is created using Seaborn's `heatmap` function and displayed using Matplotlib's `figure` method.

5. Feature engineering: The code creates two new columns: "bedroom_ratio" and "rooms_ratio", which are ratios of total bedrooms to total rooms, and total rooms to households respectively.

6. Splitting data: The code splits the data into training and testing sets using Scikit-learn's `train_test_split` function.

7. Model training: Two regression models are trained on the data. First, a linear regression model is trained using Scikit-learn's `LinearRegression` class and its `fit` method. The model's accuracy is evaluated using the `score` method on the test set. Second, a random forest regressor is trained using Scikit-learn's `RandomForestRegressor` class and its `fit` method. The model's accuracy is again evaluated using the `score` method on the test set.

8. Hyperparameter tuning: The code creates a parameter grid for the random forest regressor and uses GridSearchCV to find the best set of hyperparameters that minimize the mean squared error on the training set. The best estimator is then used to predict on the test set, and its accuracy is evaluated using the `score` method. 

Overall, this code shows an example of a typical data preparation and model training pipeline for a machine learning regression problem. It involves loading data, cleaning and visualizing it, engineering features, splitting it into training and testing sets, training models, and tuning hyperparameters to improve model performance.
