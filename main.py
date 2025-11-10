import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('tested.csv')
data.info()

print("Missing values before preprocessing:\n", data.isnull().sum())

# Data Cleaning and features Engineering
def preprocess_data(data):
    data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    data.fillna({'Embarked': 'S'}, inplace=True)

    data.drop(columns=['Embarked'], inplace=True)

    fill_missing_ages(data)

    # Convert gender
    data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})

    # Feature Engineering
    data['FamilySize'] = data['SibSp'] + data['Parch']
    data['isAlone'] = np.where(data['FamilySize'] == 0, 1, 0)

    # Handle Fare: Fill missing with median
    data.fillna({'Fare': data['Fare'].median()}, inplace=True)
    data['FareBin'] = pd.cut(data['Fare'], 4, labels=False)
    data['AgeBin'] = pd.cut(data['Age'], bins=[0, 12, 20, 40, 60, np.inf], labels=False)

    print('Data sample after preprocessing:\n', data)

    return data


# Fill in missing ages

def fill_missing_ages(data):
    age_fill_map = {}
    for pclass in data['Pclass'].unique():
        if pclass not in age_fill_map:
            age_fill_map[pclass] = data[data['Pclass'] == pclass]['Age'].median()
    
    data['Age'] = data.apply(lambda row: age_fill_map[row['Pclass']] if pd.isnull(row['Age']) else row['Age'], axis=1)

data = preprocess_data(data)
print("Missing values after preprocessing:\n", data.isnull().sum())

# Create Features / Target Variables(Make FlashCards)
x = data.drop(columns=['Survived'])
y = data['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

print('X_train::::', x_train.head())
print('X_Test::::', x_test.head())

# Machine Learning processing
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Hyperparameter tunning - KNN
def tune_model(x_train, y_train):
    param_grid = {
        "n_neighbors": range(1,21),
        "metric": ["euclidean", "manhattan", "minkowski"],
        "weights": ["uniform", "distance"]
    }

    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=1)
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_

best_model = tune_model(x_train, y_train)

# Predictions and Evaluate
def evaluate_model(model, x_test, y_test):
    prediction = model.predict(x_test)
    accuracy = accuracy_score(y_test, prediction)
    matrix = confusion_matrix(y_test, prediction)
    return accuracy, matrix

accuracy, matrix = evaluate_model(best_model, x_test, y_test)

print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Confusion matrix:')
print(matrix)

# Plot
def plot_model(matrix):
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', cbar=False, xticklabels=["Not Survived", "Survived"], yticklabels=["Not Survived", "Survived"])
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.show()

plot_model(matrix)
