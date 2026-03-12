import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

df = pd.read_csv('data/logistics_data.csv')

le_vehicle = LabelEncoder()
le_road = LabelEncoder()
df['vehicle_type'] = le_vehicle.fit_transform(df['vehicle_type'])
df['road_type'] = le_road.fit_transform(df['road_type'])

X = df[['vehicle_type', 'distance_km', 'cargo_weight_kg', 'traffic_level', 'road_type']]
y_fuel = df['fuel_consumed_liters']
y_emission = df['carbon_emission_kg']

X_train, X_test, y_fuel_train, y_fuel_test = train_test_split(X, y_fuel, test_size=0.2, random_state=42)
_, _, y_emission_train, y_emission_test = train_test_split(X, y_emission, test_size=0.2, random_state=42)

# Random Forest - route efficiency
fuel_model = RandomForestRegressor(n_estimators=100, random_state=42)
fuel_model.fit(X_train, y_fuel_train)

# Gradient Boosting - emission prediction
emission_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
emission_model.fit(X_train, y_emission_train)

# Linear Regression - fuel consumption prediction
lr_model = LinearRegression()
lr_model.fit(X_train, y_fuel_train)

os.makedirs('models', exist_ok=True)
pickle.dump(fuel_model, open('models/fuel_model.pkl', 'wb'))
pickle.dump(emission_model, open('models/emission_model.pkl', 'wb'))
pickle.dump(lr_model, open('models/lr_model.pkl', 'wb'))
pickle.dump(le_vehicle, open('models/le_vehicle.pkl', 'wb'))
pickle.dump(le_road, open('models/le_road.pkl', 'wb'))

print("✅ All 3 models trained and saved successfully!")
print(f"Random Forest R2:       {r2_score(y_fuel_test, fuel_model.predict(X_test)):.4f}")
print(f"Gradient Boosting R2:   {r2_score(y_emission_test, emission_model.predict(X_test)):.4f}")
print(f"Linear Regression R2:   {r2_score(y_fuel_test, lr_model.predict(X_test)):.4f}")
