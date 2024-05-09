import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.layers import Activation
data = pd.read_csv('AAPL.csv')
data['Date'] = pd.to_datetime(data['Date'])
prices = data['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

def create_sequences(dataset, time_steps=1):
    x, y = [], []
    for i in range(len(dataset) - time_steps - 1):
        a = dataset[i:(i + time_steps), 0]
        x.append(a)
        y.append(dataset[i + time_steps, 0])
    return np.array(x), np.array(y)

time_steps = 10
X, y = create_sequences(prices_scaled, time_steps)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

mae_scores = []
rmse_scores = []
r2_scores = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=1, activation='tanh', padding='same', input_shape=(time_steps, 1)))
    model.add(MaxPooling1D(pool_size=1, padding='same'))
    model.add(Activation('relu'))
    model.add(LSTM(units=64, activation='tanh'))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_absolute_error')
    model.fit(X_train, y_train, epochs=500, batch_size=64, verbose=0)

    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae_scores.append(mean_absolute_error(y_test_rescaled, y_pred_rescaled))
    rmse_scores.append(np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled)))
    r2_scores.append(r2_score(y_test_rescaled, y_pred_rescaled))

print(f"CNN-LSTM model:")
print("Средний MAE:", np.mean(mae_scores))
print("Средний RMSE:", np.mean(rmse_scores))
print("Средний R²:", np.mean(r2_scores))