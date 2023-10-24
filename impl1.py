import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os


# Load the dataset
absolute_path = '/Users/alan/Downloads/NVDA.csv'
df = pd.read_csv(absolute_path)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Preprocessing: Scaling the stock prices between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1,1))

# Training data: 80% of data
training_data_len = int(np.ceil(len(scaled_data) * 0.8))

# Creating a dataset with 60 previous days stock price
x_train, y_train = [], []
for i in range(60, training_data_len):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=5)

# Create a test dataset
test_data = scaled_data[training_data_len - 60:, :]
x_test, y_test = [], df['Close'][training_data_len:].values
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predicting stock prices
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Evaluate the model using RMSE
rmse = mean_squared_error(y_test, predictions, squared=False)
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Plot the test and predicted prices
plt.figure(figsize=(15,7))
plt.plot(df.index[training_data_len:], y_test, label="Actual", color='blue')
plt.plot(df.index[training_data_len:], predictions, label="Predicted", color='red')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()

# Predict future stock prices for 25 October - 7 November
future_dates = pd.date_range(start='2023-10-25', end='2023-11-07', freq='B')
future_data = scaled_data[-60:]
for i in range(len(future_dates)):
    x = future_data[-60:]
    x = x.reshape((1, 60, 1))
    pred = model.predict(x)
    future_data = np.append(future_data, pred)
    x = np.array([pred])

future_predictions = scaler.inverse_transform(future_data[-len(future_dates):].reshape(-1,1))
forecast_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['Predicted Close'])


# Define the filename
filename = "Implementation1.csv"
# Get the current working directory
current_directory = os.getcwd()
# Combine the current directory and filename to create the full path
full_path = os.path.join(current_directory, filename)

# Assuming forecast_df is your DataFrame
forecast_df.to_csv(full_path)

print("Predictions saved to Implementation1.csv!")


