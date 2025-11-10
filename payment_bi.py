import pandas as pd
import numpy as np
from pymongo import MongoClient

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt

# 1. Connect to Mongodb
client = MongoClient('mongodb://192.168.20.19:27017/?directConnection=true')
db = client['payall-education-gestion-payment']

# ---------- Define the MongDB pipeline ---------
pipeline = [
    {
        '$lookup': {
            'from': 'frais',
            'localField': 'frais',
            'foreignField': '_id',
            'as': 'fee_details'
        }
    },
    {
        '$unwind': {
            'path': '$fee_details',
            'preserveNullAndEmptyArrays': True  # Keep documents even if no matching frais
        }
    },
    {
        '$lookup': {
            'from': 'typefrais',
            'localField': 'fee_details.type_frais',
            'foreignField': '_id',
            'as': 'type_fee_details'
        }
    },
    {
        '$unwind': {
            'path': '$type_fee_details',
            'preserveNullAndEmptyArrays': True  # Keep documents even if no matching type frais
        }
    },
    {
        '$project': {
        'matricule': '$etudiant.matricule',
        'etudiant': {
            '$concat': ['$etudiant.nom', ' ', '$etudiant.prenom']
        },
        'fee type': '$type_fee_details.designation',
        'montant': '$fee_details.montant',
        'mode_paiement': '$mode_paiement.designation',
        'date_paiement': '$date_paiement',
        'montant_paye': '$montant_paye',
        'motif': '$motif',
        'devise': '$devise.code',
        'source': '$source.code',
        }
    }
]

payment_query = db.paiements.aggregate(pipeline)

# Convert the results to a list and load into a Pandas DataFrame
payment_data = list(payment_query)
df = pd.DataFrame(payment_data)

# Print the first few rows to verify
print(df.head())

print(df.isnull().sum())
df.drop(columns='_id', inplace=True)

print('============================= After Dropping unnecessary columns =============================')
print(df.isnull().sum())
print(df.head())

print("============================= Data Preprocessing =======================================")

# Define column types
numeric_columns = ['montant', 'montant_paye']  # Numeric columns
datetime_columns = ['date_paiement']   # Datetime column
categorical_columns = ['matricule', 'etudiant', 'fee type', 'motif', 'mode_paiement', 'devise', 'source']  # Categorical columns

def preprocess_data(data):
    print('data preprocessing')
    # 1. Handle numeric columns (median imputation)
    for col in numeric_columns:
        if data[col].isnull().sum() > 0:  # Only impute if missing values exist
            data[col].fillna(data[col].median(), inplace=True)

    # 2. Handle datetime column (convert to datetime, then median per 'etudiant')
    for col in datetime_columns:
        data[col] = pd.to_datetime(data[col], errors='coerce')
        
        data[col] = data.groupby('etudiant')[col].transform(
            lambda x: x.fillna(x.median())
        )
        # Fallback to overall median if group median is NaN
        overall_median = data[col].median()
        data[col] = data[col].fillna(overall_median)
        # Convert Timestamp to numeric: days since a reference date (e.g., dataset min)
        reference_date = data[col].min()
        data[f'{col}_days'] = (data[col] - reference_date).dt.days
        # Drop original datetime column to avoid Timestamp in model
        data = data.drop(columns=[col])

    # 3. Handle categorical columns (mode or 'Unknown')
    for col in categorical_columns:
        data[col] = data.groupby('etudiant')[col].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown')
        )
        # Fallback to overall mode or 'Unknown' if group mode is empty
        overall_mode = data[col].mode()[0] if not data[col].mode().empty else 'Unknown'
        data[col] = data[col].fillna(overall_mode)
    return data

dt_preprocessed = preprocess_data(df)

print("==================================== After Preprocessing Data ================================")

print(dt_preprocessed)

Encoder = OrdinalEncoder()
dt = dt_preprocessed.copy()

for encoded_label in categorical_columns:
    dt[encoded_label] = Encoder.fit_transform(dt[[encoded_label]])

# Prediction Tasks to implement
# 1. Predict Payment Delays or Defaults
# 2. Predict Payment Amounts
# 3. Predict Preferred Payment Method
# 4. Predict Payment Trends Over Time
# 5. Detect Anomalous Payments

# Splitting Data

x = dt_preprocessed.drop('montant', axis=1)
y = dt_preprocessed['montant']

X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(dt.drop('montant', axis=1), dt['montant'], test_size=0.3, random_state=101)


print(X_train_df)
print(X_test_df)

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
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train_df, y_train_df)
    y_pred = model.predict(X_test_df)
    results[name] = evaluate_model(y_test_df, y_pred, n_samples=X_test_df.shape[0], n_features=X_test_df.shape[1])

print('Results :::::>\n', results)

result_data_frame = pd.DataFrame(results).T.sort_values(by="R2", ascending=False)

print('Results of DataFrame:::::>\n', result_data_frame)

# (a) Compare Actual vs Predicted (Scatter Plot)
def plot_actual_vs_pred(y_true, y_pred, title):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.3, color="Blue")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual Payment Amounts")
    plt.ylabel("Predicted Payment Amounts")
    plt.title(f"Actual Payment vs Predicted Payment - {title}")
    plt.show()

plot_actual_vs_pred(y_test_df, y_pred, "Payment Amount")



