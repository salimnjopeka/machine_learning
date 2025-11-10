import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

import matplotlib.pyplot as plt

# Loading Dataset
data = pd.read_csv('shopping_behavior_updated.csv')
# print("Data info before dropping some columns\n\n",data.info())
data.drop(columns=["Customer ID", "Promo Code Used", "Discount Applied", "Subscription Status"], inplace=True)
# print("Data info After dropping some columns\n\n",data.info())
print(data.head())
print("Missing values before preprocessing:\n", data.isnull().sum())

# Preprocessing data
categorical_cols = ['Item Purchased', 'Category', 'Location', 'Size', 'Color', 'Season', 'Shipping Type', 'Payment Method', 'Frequency of Purchases']
numerical_cols = ['Age', 'Purchase Amount (USD)', 'Previous Purchases', 'Review Rating']

def preprocess_data(data):
    for col in numerical_cols:
        data.fillna({ col: data[col].median() }, inplace=True)
    for col in categorical_cols:
        data.fillna({ col: data[col].mode()[0] }, inplace=True)
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    return data

data = preprocess_data(data)

x = data.drop('Purchase Amount (USD)', axis=1)
y = data['Purchase Amount (USD)']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

Encoder = OrdinalEncoder()
dt = data.copy()

for col_encoded in categorical_cols:
    dt[col_encoded] = Encoder.fit_transform(dt[[col_encoded]])

# Splitting data

x = data.drop('Purchase Amount (USD)', axis=1)
y = data['Purchase Amount (USD)']

X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(dt.drop('Purchase Amount (USD)', axis=1), dt['Purchase Amount (USD)'], test_size=0.3, random_state=101)

print("X_train_df :\n", X_train_df.head())
print("X_test_df :\n", X_test_df.head())

def evaluate_model(y_true, y_pred, n_samples, n_features):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mrse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    return {
        "MAE": mae,
        "MSE": mse,
        "MRSE": mrse,
        "R2": r2,
        "ADJ_R2": adj_r2
    }

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
    model.fit(X_train_df, y_train_df)
    y_pred = model.predict(X_test_df)
    results[name] = evaluate_model(y_test_df, y_pred, n_samples=X_test_df.shape[0], n_features=X_test_df.shape[1])

print('Results :::::>\n', results)

results_df = pd.DataFrame(results).T.sort_values(by="R2", ascending=False)
print('results_df:\n', results_df)

# (a) Compare Actual vs Predicted (Scatter Plot)
def plot_actual_vs_pred(y_true, y_pred, title):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.3, color="Blue")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual Parchased Amount")
    plt.ylabel("Predicted Parchased Amount")
    plt.title(f"Actual vs Predicted - {title}")
    plt.show()

# (b) Residuals Plot
def plot_residuals(y_true, y_pred, title):
    residuals = y_true - y_pred
    plt.figure(figsize=(6,4))
    plt.scatter(y_pred, residuals, alpha=0.3, color="purple")
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Predicted Parchased Amount")
    plt.ylabel("Residuals (Error)")
    plt.title(f"Residual Plot - {title}")
    plt.show()

# (c) Compare Models (Bar Chart)
results_df = pd.DataFrame(results).T
results_df[['MAE','MRSE','R2']].plot(kind="bar", figsize=(12,6))
plt.title("Model Comparison")
plt.ylabel("Score / Error")
plt.xticks(rotation=45)
plt.show()

plot_actual_vs_pred(y_test_df, y_pred, "Best Model")
plot_residuals(y_test_df, y_pred, "Best Model")