import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import googlemaps
from datetime import datetime
from pyowm import OWM

# Google Maps API key (replace with your own key)
google_maps_api_key = 'YOUR_GOOGLE_MAPS_API_KEY'

# OpenWeatherMap API key (replace with your own key)
owm_api_key = 'YOUR_OPENWEATHERMAP_API_KEY'

def get_traffic_density(api_key, origin, destination):
    gmaps = googlemaps.Client(key=api_key)

    # Get current traffic conditions
    now = datetime.now()
    directions_result = gmaps.directions(origin, destination, departure_time=now)

    if directions_result:
        traffic_density = directions_result[0]['legs'][0]['duration_in_traffic']['value']
        return traffic_density
    else:
        return None

def get_weather_data(api_key, city):
    owm = OWM(api_key)
    mgr = owm.weather_manager()

    # Search for the city
    observation = mgr.weather_at_place(city)

    if observation:
        weather = observation.weather
        return weather.temperature('celsius')['temp'], weather.humidity
    else:
        return None

# Generate synthetic data with more samples
np.random.seed(42)
n_samples = 1000
data = {
    'Population': np.random.randint(500000, 5000000, n_samples),
    'Area': np.random.randint(30, 300, n_samples),
    'GreenSpaces': np.random.randint(10, 100, n_samples),
    'TrafficDensity': np.zeros(n_samples),  # Initialize with zeros
    'Temperature': np.zeros(n_samples),  # Initialize with zeros
    'Humidity': np.zeros(n_samples),  # Initialize with zeros
    'CO2Emission': np.random.randint(200, 1000, n_samples)
}

df = pd.DataFrame(data)

# Replace TrafficDensity, Temperature, and Humidity with real-world data
for i in range(n_samples):
    origin = 'YourOrigin'  # Replace with your actual origin
    destination = 'YourDestination'  # Replace with your actual destination
    city = 'YourCity'  # Replace with your actual city

    traffic_density = get_traffic_density(google_maps_api_key, origin, destination)

    if traffic_density is not None:
        df.at[i, 'TrafficDensity'] = traffic_density
    else:
        print('Failed to fetch traffic data.')

    temperature, humidity = get_weather_data(owm_api_key, city)

    if temperature is not None:
        df.at[i, 'Temperature'] = temperature
        df.at[i, 'Humidity'] = humidity
    else:
        print('Failed to fetch weather data.')

# Split the data into features (X) and target variable (y)
X = df.drop('CO2Emission', axis=1)
y = df['CO2Emission']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the GradientBoostingRegressor model
model = GradientBoostingRegressor(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print(f'Best Hyperparameters: {best_params}')

# Train the model with the best hyperparameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model with cross-validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)
print(f'Cross-Validation RMSE Scores: {cv_rmse_scores}')
print(f'Mean CV RMSE: {np.mean(cv_rmse_scores)}')

# Final evaluation on the test set
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Final RMSE on Test Set: {final_rmse}')
