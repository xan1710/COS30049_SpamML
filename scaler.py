from sklearn.preprocessing import StandardScaler
import numpy as np

# Sample data
data = np.array([[10, 2], [20, 4], [30, 6], [40, 8]])

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the scaler to the data and transform it
scaled_data = scaler.fit_transform(data)

print("Original Data:\n", data)
print("Scaled Data:\n", scaled_data)