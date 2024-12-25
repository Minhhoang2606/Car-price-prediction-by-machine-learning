'''
Car Price Prediction Using Machine Learning
Author: Henry Ha
'''
# Import libraries
from attr import asdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#TODO EDA

# Load the dataset
car_data = pd.read_csv('car data.csv')

# Display dataset info and summary
print(car_data.info())
print(car_data.describe())

# Plotting histograms for numerical features
numerical_features = ['Year', 'Present_Price', 'Kms_Driven', 'Selling_Price']
car_data[numerical_features].hist(figsize=(12, 8), bins=15)
plt.show()

# Count plots for categorical features as subplots
categorical_features = ['Fuel_Type', 'Seller_Type', 'Transmission']
fig, axes = plt.subplots(1, len(categorical_features), figsize=(15, 5))

for ax, feature in zip(axes, categorical_features):
    sns.countplot(data=car_data, x=feature, ax=ax)
    ax.set_title(f'Distribution of {feature}')

plt.tight_layout()
plt.show()

# Scatter plots for relationships
sns.pairplot(car_data, vars=numerical_features, hue='Fuel_Type')
plt.show()

# Selecting numerical features
numerical_features = car_data.select_dtypes(include=['int64', 'float64'])

# Compute correlation matrix
corr_matrix = numerical_features.corr()

# Visualize the correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap (Numerical Features)')
plt.show()

#TODO Data preprocessing

# Dropping unnecessary columns
car_data.drop(['Car_Name'], axis=1, inplace=True)

# Converting Year to Age
import datetime
current_year = datetime.datetime.now().year
car_data['Age'] = current_year - car_data['Year']
car_data.drop(['Year'], axis=1, inplace=True)

# Encoding categorical features
car_data = pd.get_dummies(car_data, drop_first=True)

# Splitting the dataset
X = car_data.drop('Selling_Price', axis=1)
y = car_data['Selling_Price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train, X_test)

#TODO Model building

# Linear Regression Model
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred_lr = linear_model.predict(X_test)

# Decision Tree Model
from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred_dt = dt_model.predict(X_test)

# Random Forest Model
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred_rf = rf_model.predict(X_test)

#TODO Model evaluation

from sklearn.metrics import r2_score, mean_squared_error

# Function to evaluate a model
def evaluate_model(model_name, y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{model_name}:")
    print(f"R² Score: {r2:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print("-" * 30)

# Evaluate Linear Regression
evaluate_model("Linear Regression", y_test, y_pred_lr)

# Evaluate Decision Tree Regressor
evaluate_model("Decision Tree", y_test, y_pred_dt)

# Evaluate Random Forest Regressor
evaluate_model("Random Forest", y_test, y_pred_rf)

#TODO Fine-Tuning the Model

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the Random Forest model
rf = RandomForestRegressor(random_state=42)

# Perform Grid Search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           scoring='r2', cv=3, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters
print("Best Parameters:", grid_search.best_params_)

# Train the fine-tuned model
best_rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
best_rf.fit(X_train, y_train)

# Make predictions
y_pred_best_rf = best_rf.predict(X_test)

# Evaluate the fine-tuned model
r2 = r2_score(y_test, y_pred_best_rf)
mse = mean_squared_error(y_test, y_pred_best_rf)

print(f"Fine-Tuned Random Forest:")
print(f"R² Score: {r2:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

import joblib

# Save the trained Random Forest model
joblib.dump(rf_model, 'random_forest_model.pkl')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

joblib.dump(X.columns.tolist(), 'features.pkl')