
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("./dataset/real_estate_dataset.csv")

X = df[
    [
        "House_Size_sqft",
        "Bedrooms",
        "House_Age_Years",
        "Distance_to_City_km",
        "Nearby_Schools"
    ]
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


sigmoid_output = 1 / (1 + np.exp(-X_scaled))


tanh_output = np.tanh(X_scaled)

sigmoid_df = pd.DataFrame(
    sigmoid_output,
    columns=X.columns
)

tanh_df = pd.DataFrame(
    tanh_output,
    columns=X.columns
)

print("ORIGINAL DATA:")
print(X.head())

print("\nSIGMOID OUTPUT:")
print(sigmoid_df.head().round(3))

print("\nTANH OUTPUT:")
print(tanh_df.head().round(3))