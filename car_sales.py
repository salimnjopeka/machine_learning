import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb


import matplotlib.pyplot as plt
import seaborn as sns

# Load and inspect data
data = pd.read_csv('car_sales_data.csv')
data.info()
print("Missing values before preprocessing:\n", data.isnull().sum())


# Basic cleaning (handle missing values; adjust strategies as needed)
# Fill numerical with median, categorical with mode
numerical_cols = ['Engine size', 'Year of manufacture', 'Mileage', 'Price']
categorical_cols = ['Manufacturer', 'Model', 'Fuel type']


# Preprocessing data
def preprocess_data(data):
    for col in numerical_cols:
        data.fillna({col: data[col].median()}, inplace=True)
    for col in categorical_cols:
        data.fillna({col: data[col].mode()[0]}, inplace=True)
    return data

preprocess_data(data)
print("Missing values after preprocessing:\n", data.isnull().sum())


x = data.drop('Price', axis=1)
y = data['Price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

print('x_train table:\n', x_train)
print('x_test table:\n', x_test)

Encoder = OrdinalEncoder()
dt = data.copy()

for col in ['Manufacturer','Fuel type','Model']:
    dt[col]= Encoder.fit_transform(dt[[col]])

x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(dt.drop('Price',axis=1), dt['Price'], test_size=0.3, random_state=101)

print('x_train_df=============>:\n', x_train_df)
print('x_test_df=============>:\n', x_test_df)

def evaluate_model(y_true, y_pred, n_samples, n_features):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    return { "mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "adj_r2": adj_r2 }


models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
}

results = {}

for name, model in models.items():
    # Train model
    model.fit(x_train_df, y_train_df)

    # Predictions
    y_pred = model.predict(x_test_df)

    # Evaluate
    results[name] = evaluate_model(y_test_df, y_pred, n_samples=x_test_df.shape[0], n_features=x_test_df.shape[1])

def convert(o):
    if isinstance(o, np.generic):  # covers np.float64, np.int64 etc.
        return o.item()
    raise TypeError

# Pretty-print JSON
print(json.dumps(results, indent=2, default=convert))
results_df = pd.DataFrame(results).T.sort_values(by="r2", ascending=False)
print('results_df:\n', results_df)

best_model = results_df.index[0]
best_r2 = results_df.loc[best_model, 'r2']
print(f"\nBest Model: {best_model}")
print(f"Best R² Score: {best_r2:.4f}")

# (a) Compare Actual vs Predicted (Scatter Plot)
def plot_actual_vs_pred(y_true, y_pred, title):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.3, color="Blue")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Actual vs Predicted - {title}")
    plt.show()

# plot_actual_vs_pred(y+x_test_df, y_pred, "Random Forest")

# (b) Residuals Plot
# def plot_residuals(y_true, y_pred, title):
#     residuals = y_true - y_pred
#     plt.figure(figsize=(6,4))
#     plt.scatter(y_pred, residuals, alpha=0.3, color="purple")
#     plt.axhline(y=0, color="r", linestyle="--")
#     plt.xlabel("Predicted Price")
#     plt.ylabel("Residuals (Error)")
#     plt.title(f"Residual Plot - {title}")
#     plt.show()

# (c) Compare Models (Bar Chart)
# results_df = pd.DataFrame(results).T
# results_df[['mae','rmse','r2']].plot(kind="bar", figsize=(12,6))
# plt.title("Model Comparison")
# plt.ylabel("Score / Error")
# plt.xticks(rotation=45)
# plt.show()


