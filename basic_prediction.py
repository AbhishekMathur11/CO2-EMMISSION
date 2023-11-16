# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate synthetic data
data = {
    'Population': [500000, 700000, 1000000, 1500000, 2000000],
    'Area': [50, 70, 100, 150, 200],
    'GreenSpaces': [20, 30, 40, 50, 60],
    'TrafficDensity': [2000, 1800, 1500, 1200, 1000],
    'CO2Emission': [300, 400, 500, 600, 700]
}

df = pd.DataFrame(data)

# Split the data into features (X) and target variable (y)
X = df.drop('CO2Emission', axis=1)
y = df['CO2Emission']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize the results
plt.scatter(X_test['Population'], y_test, color='black', label='Actual')
plt.scatter(X_test['Population'], y_pred, color='blue', label='Predicted')
plt.xlabel('Population')
plt.ylabel('CO2 Emission')
plt.legend()
plt.show()
