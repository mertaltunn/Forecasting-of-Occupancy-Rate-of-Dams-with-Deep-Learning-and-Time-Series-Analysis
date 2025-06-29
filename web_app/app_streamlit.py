# Importing necessary libraries
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import numpy as np

# Dynamic path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, '..', 'data', 'raw', 'istanbul-dams-daily-occupancy-rates.xlsx')

# Loading and cleaning the dataset
df = pd.read_excel(data_path)
df['Tarih'] = pd.to_datetime(df['Tarih'], dayfirst=True)
for col in df.columns:
    if col != 'Tarih':
        df[col] = df[col].astype(str).str.replace(',', '.').str.replace('%', '').astype(float)
df = df.set_index('Tarih')
df.ffill(inplace=True)

# Calculating Istanbul General after cleaning
df['Istanbul General'] = df.mean(axis=1)

# Forecast functions with caching
@st.cache_resource
def forecast_with_prophet_cached(dam_series, forecast_days):
    dam_df = dam_series.reset_index()
    dam_df.columns = ['ds', 'y']
    model = Prophet()
    model.fit(dam_df)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(forecast_days)

@st.cache_resource
def forecast_with_sarima_cached(dam_series, forecast_days):
    dam_series = dam_series.asfreq('D')
    model = sm.tsa.statespace.SARIMAX(dam_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    result = model.fit(disp=False)
    forecast = result.get_forecast(steps=forecast_days)
    pred_mean = forecast.predicted_mean
    return pred_mean

@st.cache_resource
def forecast_with_lstm_cached(dam_series, forecast_days, look_back=30):
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(dam_series.values.reshape(-1, 1))
    X, y = [], []
    for i in range(look_back, len(series_scaled)):
        X.append(series_scaled[i - look_back:i])
        y.append(series_scaled[i])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, batch_size=16, verbose=0)

    last_seq = series_scaled[-look_back:]
    preds = []
    current_seq = last_seq
    for _ in range(forecast_days):
        pred = model.predict(current_seq.reshape(1, look_back, 1), verbose=0)
        preds.append(pred[0, 0])
        current_seq = np.append(current_seq[1:], pred).reshape(look_back, 1)

    forecast = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    forecast_dates = pd.date_range(dam_series.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_series = pd.Series(forecast, index=forecast_dates)
    return forecast_series

# Streamlit App UI
st.set_page_config(page_title="Istanbul Dams Forecasting", layout="wide", initial_sidebar_state="expanded")

st.sidebar.title("Forecasting Dashboard")
selected_dam = st.sidebar.selectbox("Select Dam", list(df.columns))
forecast_days = st.sidebar.slider("Forecast Days", 7, 365, 20)
selected_model = st.sidebar.selectbox("Select Model", ["Prophet", "SARIMA", "LSTM"])

# Historical Data Visualization (optimized)
st.title("Istanbul Dams Forecasting")
st.subheader(f"Selected Dam: {selected_dam}")
fig, ax = plt.subplots(figsize=(14, 5))
dam_series_clean = df[selected_dam].asfreq('D')
dam_series_clean = dam_series_clean.loc[dam_series_clean.first_valid_index():dam_series_clean.last_valid_index()]
ax.plot(dam_series_clean.index, dam_series_clean.values, label='Historical Data', color='steelblue')
ax.set_title(f"{selected_dam} Historical Occupancy")
ax.set_xlabel("Date")
ax.set_ylabel("Occupancy Rate (%)")
ax.legend()
ax.grid(True)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)
st.pyplot(fig)

# Forecasting (with clear separation from graph)
st.subheader(f"Forecast for next {forecast_days} days using {selected_model}")
with st.spinner(f"Generating {selected_model} forecast..."):
    if selected_model == "Prophet":
        forecast_df = forecast_with_prophet_cached(dam_series_clean, forecast_days)
        fig2, ax2 = plt.subplots(figsize=(14, 5))
        ax2.plot(forecast_df['ds'], forecast_df['yhat'], label='Prophet Forecast', marker='o')
    elif selected_model == "SARIMA":
        forecast_series = forecast_with_sarima_cached(dam_series_clean, forecast_days)
        fig2, ax2 = plt.subplots(figsize=(14, 5))
        ax2.plot(forecast_series.index, forecast_series.values, label='SARIMA Forecast', marker='o')
    else:
        forecast_series = forecast_with_lstm_cached(dam_series_clean, forecast_days)
        fig2, ax2 = plt.subplots(figsize=(14, 5))
        ax2.plot(forecast_series.index, forecast_series.values, label='LSTM Forecast', marker='o')

    ax2.set_title(f"{selected_dam} Forecast ({selected_model})")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Occupancy Rate (%)")
    ax2.legend()
    ax2.grid(True)
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, forecast_days // 5)))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    st.pyplot(fig2)
