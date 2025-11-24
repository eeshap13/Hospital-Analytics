#predicts length of stay using a synthetic sample patient dataset

import pandas as pd
import numpy as np
import os
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import clean_data, encode_categoricals

#directories that store trained models and evaluation metrics
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

#generating sample data
np.random.seed(42) 
n = 1000 #number of patients

df = pd.DataFrame({
    "age": np.random.randint(20, 90, size=n),  #age of patient
    "sex": np.random.choice(["M", "F"], size=n),  #sex of patient
    "blood_pressure": np.random.randint(90, 180, size=n),  #Systolic BP
    "glucose": np.random.randint(70, 200, size=n),  #blood glucose
    "comorbidities": np.random.randint(0, 5, size=n),  #number of comorbidities
    "previous_visits": np.random.randint(0, 10, size=n),  #prior hospital visits
})

#making a linear relationship with some noise to analyze the target: length of stay (in days)
df["length_of_stay"] = (
    2 + 0.05*df["age"] + 0.1*df["comorbidities"] +
    0.02*df["blood_pressure"] + np.random.normal(0, 1, n)
).round() #gives data to nearest day

df["sex"] = df["sex"].map({"M": 1, "F": 0}) #for numerical analysis in heatmap

#saving dataset locally (can upload another file)
df.to_csv("data/LengthOfStay.csv", index=False)

#use preprocessing functions from utils.py
#clean_data(): fills missing values and drops duplicates
#encode_categoricals(): converts categorical columns to numeric
df = clean_data(df)
df = encode_categoricals(df)

target = "length_of_stay"
#columns except target columns are features
features = [col for col in df.columns if col != target]

X = df[features]  #feature matrix
y = df[target]    #target vector

#using 80% of data for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#StandardScaler: mean=0, std=1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  #fitting on training set
X_test_scaled = scaler.transform(X_test)        #applying same transformation to test set

#saving scaler for future use
joblib.dump(scaler, "models/scaler.pkl")

#linear regression as baseline
los_model = LinearRegression()
los_model.fit(X_train_scaled, y_train)

#saving trained model
joblib.dump(los_model, "models/los_model.pkl")

#evaluating model
y_pred = los_model.predict(X_test_scaled)

metrics = {
    "MAE": float(mean_absolute_error(y_test, y_pred)),  #mean absolute error
    "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred)))  #root mean squared error
}

#saving metrics to json file
with open("results/los_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

#printing metrics to console
print("Length of Stay Regression Metrics:")
print(metrics)
