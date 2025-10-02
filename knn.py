from sklearn.neighbors import KNeighborsRegressor
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 6])

# Initialize and train the regressor
knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X, y)

# Make a prediction
prediction = knn_reg.predict(np.array([[3.5]]))
print("Prediction for input 3.5:", prediction)