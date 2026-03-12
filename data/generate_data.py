import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 1000

vehicle_types = ['bike', 'car', 'truck', 'van']
vehicle_emission_factor = {'bike': 0.05, 'car': 0.21, 'truck': 0.55, 'van': 0.32}

data = []

for _ in range(n):
    vehicle = np.random.choice(vehicle_types)
    distance = round(np.random.uniform(5, 500), 2)
    cargo_weight = round(np.random.uniform(10, 5000), 2)
    traffic_level = round(np.random.uniform(0.5, 2.0), 2)
    road_type = np.random.choice(['highway', 'city', 'mixed'])
    road_factor = {'highway': 0.8, 'city': 1.3, 'mixed': 1.0}

    fuel_consumed = round(
        (distance * cargo_weight * traffic_level * road_factor[road_type]) / 10000, 2
    )
    emission = round(fuel_consumed * vehicle_emission_factor[vehicle], 4)

    data.append([vehicle, distance, cargo_weight, traffic_level, road_type, fuel_consumed, emission])

df = pd.DataFrame(data, columns=[
    'vehicle_type', 'distance_km', 'cargo_weight_kg',
    'traffic_level', 'road_type', 'fuel_consumed_liters', 'carbon_emission_kg'
])

os.makedirs('data', exist_ok=True)
df.to_csv('data/logistics_data.csv', index=False)
print("✅ Dataset created successfully! Total records:", len(df))
print(df.head())
