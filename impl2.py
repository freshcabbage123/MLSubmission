import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
import os

# Load the dataset
absolute_path = '/Users/alan/Downloads/NVDA.csv'
df = pd.read_csv(absolute_path)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.resample('B').ffill()

# Visualize the data
df['Close'].plot(figsize=(15, 7))
plt.title('Closing Price Over Time')
plt.show()

# Check for stationarity using the ADF test
result = adfuller(df['Close'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# If not stationary, difference the series
if result[1] > 0.05:
    df['Close_diff'] = df['Close'].diff()
    df.dropna(inplace=True)

# Determine ARIMA parameters using ACF and PACF plots
plot_acf(df['Close'])
plt.show()

plot_pacf(df['Close'])
plt.show()

# Based on the plots, assign values to p, d, and q
p = 1
d = 1
q = 1

# Split data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df['Close'][:train_size], df['Close'][train_size:]

# Build and fit the ARIMA model
model = ARIMA(train, order=(p, d, q))
model_fit = model.fit()

# Predict values for the test set
forecast = model_fit.forecast(steps=len(test)).values

# Calculate and print RMSE
rmse = mean_squared_error(test, forecast, squared=False)
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Plot the actual vs. predicted values
plt.figure(figsize=(15, 7))
plt.plot(test.index, test, label="Actual", color='blue')
plt.plot(test.index, forecast, label="Forecast", color='red')
plt.title('Actual vs Forecasted Prices')
plt.legend()
plt.show()

# Forecast the future values for the period 25 October 2023 to 7 November 2023
forecast_dates = pd.date_range(start="2023-10-25", end="2023-11-07", freq='B')
future_forecast = model_fit.forecast(steps=len(forecast_dates))

# Convert forecasted data into a DataFrame
forecast_df = pd.DataFrame({
    'Predicted_Close': future_forecast.values
}, index = forecast_dates)

# Print the future forecasts
print(forecast_df)
# Define the filename
filename = "Implementation2.csv"
# Get the current working directory
current_directory = os.getcwd()
# Combine the current directory and filename to create the full path
full_path = os.path.join(current_directory, filename)

# Assuming forecast_df is your DataFrame
forecast_df.to_csv(full_path)

print("Predictions saved to Implementation2.csv!")
