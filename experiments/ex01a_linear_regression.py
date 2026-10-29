import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("../MLdatasets/headbrain.csv")
print(data)

X = data["Head Size(cm^3)"]
y = data["Brain Weight(grams)"]

# Split into training and testing sets (33% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(np.expand_dims(X_train, axis=1), np.expand_dims(y_train, axis=1))
predictions = model.predict(np.expand_dims(X_test, axis=1))

# Plot scatter points and regression line
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color="r", label='Scatter Plot')
plt.plot(X_test, predictions, color="b", label='Regression Line')
plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()

# Error Estimation (RMSE & R2 Score)
r2_score = model.score(np.expand_dims(X_test, axis=1), np.expand_dims(y_test, axis=1))
mse = mean_squared_error(y_test, predictions)
print(f"RMSE VALUE : {np.sqrt(mse)}\nR2 SCORE : {r2_score}")
