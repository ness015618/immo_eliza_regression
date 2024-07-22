import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


data = pd.read_csv("clean_data.csv")

# Removing outliers
Q1 = data["Price"].quantile(0.25)
Q3 = data["Price"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - (1.5*IQR)
upper_bound = Q3 + (1.5*IQR)
data = data[(data["Price"] >= lower_bound) & (data["Price"] <= upper_bound)]

# Splitting the dataset
X = data[["BedroomCount","TypeOfProperty","SubtypeOfProperty_num","Region_num","Province_num","StateOfBuilding_num","Kitchen_num","PEB_num","TypeOfSale_num","LivingArea","GardenArea","SurfaceOfPlot","Locality_num"]]
y = data["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Testing a model
## GradientBooster
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

with open('model.pkl','wb') as file:
    pickle.dump(model, file)

y_pred = model.predict(X_test)
mae_gb = mean_absolute_error(y_test, y_pred)
print("mae_gb :", mae_gb)