import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv('AAPL.csv')
data['Date'] = pd.to_datetime(data['Date'])

# Извлечение столбца с ценами закрытия
prices = data['Close'].values.reshape(-1, 1)

# Масштабирование данных
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

# Функция для создания временных последовательностей
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

# Параметры для k-fold кросс-валидации
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

mae_scores = []
rmse_scores = []
r2_scores = []

# Кросс-валидация
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Создание и обучение модели LSTM
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(time_steps, 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    # Получение прогнозов и обратное масштабирование
    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Расчет метрик
    mae_scores.append(mean_absolute_error(y_test_rescaled, y_pred_rescaled))
    rmse_scores.append(np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled)))
    r2_scores.append(r2_score(y_test_rescaled, y_pred_rescaled))

# Вывод средних значений метрик
print(f"LSTM basic model:")
print("Средний MAE:", np.mean(mae_scores))
print("Средний RMSE:", np.mean(rmse_scores))
print("Средний R²:", np.mean(r2_scores))