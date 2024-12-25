import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from datetime import datetime
import pickle


def train_and_save_model(dataframe):
    df2 = dataframe[["Date","Sub_metering_3"]]
    df2["Date"] = pd.to_datetime(df2["Date"], format="mixed")
    df2["Month"] = df2["Date"].dt.month
    df2["Day"] = df2["Date"].dt.day
    df2.drop("Date",axis=1, inplace=True)
    df2 = df2[ df2["Sub_metering_3"]  > 0.0 ]
    X = df2[["Month","Day"]]
    y = df2["Sub_metering_3"]
    X_train, X_test, y_train, y_test = train_test_split(X ,y ,test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    with open('random_forest_model_per_day.pkl', 'wb') as file:
        pickle.dump(model, file)


def read_model_and_predict(data):
    # Load the model from pickle file
    with open('random_forest_model_per_day.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    # [[ month, day]]
    prediction = loaded_model.predict(data)
    return prediction[0]

