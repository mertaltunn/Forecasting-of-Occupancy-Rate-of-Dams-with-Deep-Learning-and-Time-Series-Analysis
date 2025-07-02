import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import os
import re
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf # TensorFlow kütüphanesi eklendi

# --- 1. Constants and Path Definitions ---
# Base directories for models and data
MODELS_BASE_DIR = "models"
RESULTS_DIR = "results"
DATA_DIR = "data/processed"

# Specific model directories
GENERAL_DAM_MODELS_DIR = os.path.join(MODELS_BASE_DIR, "general_dam")
ALL_DAMS_SYSTEM_WIDE_MODELS_DIR = os.path.join(MODELS_BASE_DIR, "all_dams_system_wide")

# Data file paths
OCCUPANCY_SYNTHETIC_PATH = os.path.join(DATA_DIR, "istanbul-dams-daily-occupancy-rates-cleaned_with_synthetic.csv")
RAINFALL_CONSUMPTION_SYNTHETIC_PATH = os.path.join(DATA_DIR, "istanbul-barajlarnda-yagis-ve-gunluk-tuketim-verileri_with_synthetic.csv")
MERGED_DAM_SPECIFIC_EXTENDED_PATH = os.path.join(DATA_DIR, "merged_dam_specific_extended.csv")

# Model selection results path
MODEL_SELECTION_RESULTS_PATH = os.path.join(RESULTS_DIR, "model_selection", "all_model_metrics.csv")
BEST_MODEL_PER_DAM_PATH = os.path.join(RESULTS_DIR, "model_selection", "best_model_per_dam.csv")
BEST_2_MODELS_PER_DAM_PATH = os.path.join(RESULTS_DIR, "model_selection", "best_2_models_per_dam.csv")

# --- 2. Data Loading and Caching ---
@st.cache_data
def load_all_data():
    """Loads all necessary datasets and model selection results."""
    st.info("Loading datasets and model selection results...")
    
    # Load extended occupancy data
    df_occupancy = pd.read_csv(OCCUPANCY_SYNTHETIC_PATH)
    df_occupancy['Tarih'] = pd.to_datetime(df_occupancy['Tarih'])
    df_occupancy.ffill(inplace=True)
    df_occupancy.bfill(inplace=True)

    # Load extended merged dam-specific data (for multivariate models with extra inputs)
    df_merged_specific = pd.read_csv(MERGED_DAM_SPECIFIC_EXTENDED_PATH)
    df_merged_specific['Tarih'] = pd.to_datetime(df_merged_specific['Tarih'])
    df_merged_specific.ffill(inplace=True)
    df_merged_specific.bfill(inplace=True)
    df_merged_specific.dropna(inplace=True) # Ensure no NaNs after ffill/bfill for sequence creation

    # Load extended rainfall/consumption data (for future forecast regressors)
    df_rainfall_consumption = pd.read_csv(RAINFALL_CONSUMPTION_SYNTHETIC_PATH)
    df_rainfall_consumption['Tarih'] = pd.to_datetime(df_rainfall_consumption['Tarih'])
    df_rainfall_consumption.set_index('Tarih', inplace=True)
    
    # Load model selection results
    df_all_metrics = pd.read_csv(MODEL_SELECTION_RESULTS_PATH)
    df_best_per_dam = pd.read_csv(BEST_MODEL_PER_DAM_PATH)
    df_best_2_per_dam = pd.read_csv(BEST_2_MODELS_PER_DAM_PATH)

    st.success("Datasets and model selection results loaded.")
    return df_occupancy, df_merged_specific, df_rainfall_consumption, df_all_metrics, df_best_per_dam, df_best_2_per_dam

# --- 3. Model Loading and Forecasting Functions ---

@st.cache_resource
def load_and_forecast_model(dam_name, model_type, forecast_days, df_occupancy, df_merged_specific, df_rainfall_consumption):
    """
    Dynamically loads the specified model and performs forecasting.
    Returns (forecast_series, actual_series, model_metrics, plot_title, forecast_lower, forecast_upper, forecast_dates_for_ci).
    """
    st.info(f"Loading and forecasting with {model_type} for {dam_name}...")

    model = None
    scaler_X = None
    scaler_y = None
    model_path = None
    scaler_X_path = None
    scaler_y_path = None
    current_df = None
    target_col_name = None
    time_steps = 60 

    # --- Model Type A: Univariate_LSTM_Only_Occupancy ---
    if model_type == "Univariate_LSTM_Only_Occupancy":
        model_path = os.path.join(MODELS_BASE_DIR, dam_name.lower(), "lstm_univariate_only_occupancy_model.h5")
        scaler_path = os.path.join(MODELS_BASE_DIR, dam_name.lower(), "scaler_univariate_only_occupancy.pkl")
        
        current_df = df_occupancy.copy()
        target_col_name = dam_name # e.g., 'Omerli'
        
        scaler = joblib.load(scaler_path)
        model = load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

        series = current_df[[target_col_name]].values
        series_scaled = scaler.transform(series)
        
        def create_sequences_univariate_lstm(data, time_steps):
            Xs, ys = [], []
            for i in range(len(data) - time_steps):
                Xs.append(data[i:(i + time_steps)])
                ys.append(data[i + time_steps])
            return np.array(Xs), np.array(ys)

        X_seq_full, y_seq_full = create_sequences_univariate_lstm(series_scaled, time_steps)
        last_sequence_scaled_input = X_seq_full[-1].reshape(1, time_steps, 1)

        future_preds_scaled = []
        for _ in range(forecast_days):
            next_pred_scaled = model.predict(last_sequence_scaled_input, verbose=0)[0,0]
            future_preds_scaled.append(next_pred_scaled)
            next_input_feature_vector_scaled = np.array([next_pred_scaled]).reshape(1,1,1)
            last_sequence_scaled_input = np.append(last_sequence_scaled_input[:, 1:, :], next_input_feature_vector_scaled, axis=1)
        
        forecast_values_inv = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()
        actual_series_for_plot = current_df[['Tarih', target_col_name]]
        forecast_lower = None
        forecast_upper = None
        forecast_dates_for_ci = None

    # --- Model Type B: Single_Dam_Multivariate_LSTM_Extra_Inputs ---
    elif model_type == "Single_Dam_Multivariate_LSTM_Extra_Inputs":
        model_path = os.path.join(MODELS_BASE_DIR, dam_name.lower(), "lstm_multivariate_extra_inputs_model.h5")
        scaler_X_path = os.path.join(MODELS_BASE_DIR, dam_name.lower(), "scaler_X_multivariate_extra_inputs.pkl")
        scaler_y_path = os.path.join(MODELS_BASE_DIR, dam_name.lower(), "scaler_y_multivariate_extra_inputs.pkl")

        current_df = df_merged_specific.copy()
        target_col_name = f"{dam_name}_Fill"
        rainfall_col_for_dam = f"{dam_name}_Rainfall"
        consumption_col_name_base = 'Istanbul_Daily_Consumption'

        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
        model = load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

        base_features_for_dam = [target_col_name, rainfall_col_for_dam, consumption_col_name_base]
        df_current_dam_model = current_df[base_features_for_dam].copy()
        df_current_dam_model[f'{target_col_name}_lag1'] = df_current_dam_model[target_col_name].shift(1)
        df_current_dam_model[f'{target_col_name}_rolling7'] = df_current_dam_model[target_col_name].rolling(window=7).mean()
        df_current_dam_model[f'{target_col_name}_rolling30'] = df_current_dam_model[target_col_name].rolling(window=30).mean()
        df_current_dam_model_cleaned = df_current_dam_model.dropna().reset_index(drop=True)

        model_features_extended = [
            target_col_name, rainfall_col_for_dam, consumption_col_name_base,
            f'{target_col_name}_lag1', f'{target_col_name}_rolling7', f'{target_col_name}_rolling30'
        ]
        
        X_scaled_full = scaler_X.transform(df_current_dam_model_cleaned[model_features_extended])
        last_sequence_scaled_input = X_scaled_full[-time_steps:].reshape(1, time_steps, X_scaled_full.shape[1])

        avg_rainfall_unscaled_for_this_dam = df_rainfall_consumption[rainfall_col_for_dam.replace('_Rainfall','')].mean()
        avg_consumption_unscaled = df_rainfall_consumption[consumption_col_name_base].mean()

        dummy_avg_features_unscaled = np.zeros((1, len(model_features_extended)))
        dummy_avg_features_unscaled[0, model_features_extended.index(rainfall_col_for_dam)] = avg_rainfall_unscaled_for_this_dam
        dummy_avg_features_unscaled[0, model_features_extended.index(consumption_col_name_base)] = avg_consumption_unscaled
        scaled_avg_dummy_features = scaler_X.transform(dummy_avg_features_unscaled)
        
        scaled_avg_rainfall = scaled_avg_dummy_features[0, model_features_extended.index(rainfall_col_for_dam)]
        scaled_avg_consumption = scaled_avg_dummy_features[0, model_features_extended.index(consumption_col_name_base)]

        future_preds_scaled = []
        for _ in range(forecast_days):
            next_pred_occupancy_scaled = model.predict(last_sequence_scaled_input, verbose=0)[0,0]
            future_preds_scaled.append(next_pred_occupancy_scaled)
            
            new_feature_vector_scaled = np.zeros(len(model_features_extended))
            new_feature_vector_scaled[model_features_extended.index(target_col_name)] = next_pred_occupancy_scaled
            new_feature_vector_scaled[model_features_extended.index(f'{target_col_name}_lag1')] = next_pred_occupancy_scaled
            new_feature_vector_scaled[model_features_extended.index(f'{target_col_name}_rolling7')] = next_pred_occupancy_scaled
            new_feature_vector_scaled[model_features_extended.index(f'{target_col_name}_rolling30')] = next_pred_occupancy_scaled
            new_feature_vector_scaled[model_features_extended.index(rainfall_col_for_dam)] = scaled_avg_rainfall
            new_feature_vector_scaled[model_features_extended.index(consumption_col_name_base)] = scaled_avg_consumption
            
            last_sequence_scaled_input = np.append(last_sequence_scaled_input[:, 1:, :], new_feature_vector_scaled.reshape(1, 1, -1), axis=1)
        
        forecast_values_inv = scaler_y.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()
        actual_series_for_plot = df_merged_specific[['Tarih', target_col_name]]
        forecast_lower = None
        forecast_upper = None
        forecast_dates_for_ci = None

    # --- Model Type C: All_Dams_Multivariate_LSTM ---
    elif model_type == "All_Dams_Multivariate_LSTM":
        model_path = os.path.join(ALL_DAMS_SYSTEM_WIDE_MODELS_DIR, "lstm_all_dams_multivariate_model.h5")
        scaler_X_path = os.path.join(ALL_DAMS_SYSTEM_WIDE_MODELS_DIR, "scaler_X_all_dams_multivariate.pkl")

        current_df = df_merged_specific.copy()
        target_col_name = f"{dam_name}_Fill"
        
        scaler_X = joblib.load(scaler_X_path)
        model = load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

        dam_fill_columns_all = [col for col in current_df.columns if col.endswith('_Fill')]
        rainfall_columns_all = [col for col in current_df.columns if col.endswith('_Rainfall')]
        consumption_column_base = 'Istanbul_Daily_Consumption'
        all_features_model_C = dam_fill_columns_all + rainfall_columns_all + [consumption_column_base]
        
        X_scaled_full = scaler_X.transform(current_df[all_features_model_C])
        last_sequence_scaled_input = X_scaled_full[-time_steps:].reshape(1, time_steps, X_scaled_full.shape[1])

        avg_rainfall_values_unscaled = df_rainfall_consumption[[col.replace('_Rainfall','') for col in rainfall_columns_all]].mean().values
        avg_consumption_unscaled = df_rainfall_consumption[consumption_column_base].mean()

        dummy_avg_features_unscaled = np.zeros((1, len(all_features_model_C)))
        for i, r_col in enumerate(rainfall_columns_all):
            dummy_avg_features_unscaled[0, all_features_model_C.index(r_col)] = avg_rainfall_values_unscaled[i]
        dummy_avg_features_unscaled[0, all_features_model_C.index(consumption_column_base)] = avg_consumption_unscaled
        scaled_avg_dummy_features = scaler_X.transform(dummy_avg_features_unscaled)
        
        scaled_avg_rainfall_values = scaled_avg_dummy_features[0, [all_features_model_C.index(col) for col in rainfall_columns_all]]
        scaled_avg_consumption_value = scaled_avg_dummy_features[0, all_features_model_C.index(consumption_column_base)]

        future_preds_scaled_all_dams = []
        for _ in range(forecast_days):
            pred_scaled_all_dams = model.predict(last_sequence_scaled_input, verbose=0)[0]
            future_preds_scaled_all_dams.append(pred_scaled_all_dams)

            new_input_row_scaled = np.zeros(len(all_features_model_C))
            target_col_indices_C = [all_features_model_C.index(col) for col in dam_fill_columns_all]
            new_input_row_scaled[target_col_indices_C] = pred_scaled_all_dams
            
            for i, r_col in enumerate(rainfall_columns_all):
                new_input_row_scaled[all_features_model_C.index(r_col)] = scaled_avg_rainfall_values[i]
            new_input_row_scaled[all_features_model_C.index(consumption_column_base)] = scaled_avg_consumption_value
            
            last_sequence_scaled_input = np.append(last_sequence_scaled_input[:, 1:, :], new_input_row_scaled.reshape(1, 1, -1), axis=1)
        
        future_preds_scaled_all_dams_array = np.array(future_preds_scaled_all_dams)
        full_future_scaled_data = np.zeros((forecast_days, len(all_features_model_C)))
        full_future_scaled_data[:, target_col_indices_C] = future_preds_scaled_all_dams_array
        
        for i, r_col in enumerate(rainfall_columns_all):
            full_future_scaled_data[:, all_features_model_C.index(r_col)] = scaled_avg_rainfall_values[i]
        full_future_scaled_data[:, all_features_model_C.index(consumption_column_base)] = scaled_avg_consumption_value
        
        future_preds_inv_all_dams = scaler_X.inverse_transform(full_future_scaled_data)
        
        target_dam_index_in_all_dams = all_features_model_C.index(target_col_name)
        forecast_values_inv = future_preds_inv_all_dams[:, target_dam_index_in_all_dams]
        actual_series_for_plot = df_merged_specific[['Tarih', target_col_name]]
        
        forecast_lower = None
        forecast_upper = None
        forecast_dates_for_ci = None

    # --- Model Type D: Multivariate_LSTM_Occupancy_Only ---
    elif model_type == "Multivariate_LSTM_Occupancy_Only":
        model_path = os.path.join(MODELS_BASE_DIR, dam_name.lower(), "lstm_multivariate_occupancy_only_model.h5")
        scaler_X_path = os.path.join(MODELS_BASE_DIR, dam_name.lower(), "scaler_X_multivariate_occupancy_only.pkl")
        scaler_y_path = os.path.join(MODELS_BASE_DIR, dam_name.lower(), "scaler_y_multivariate_occupancy_only.pkl")

        current_df = df_occupancy.copy()
        target_col_name = dam_name
        
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
        model = load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

        all_dam_occupancy_columns_base = [col for col in current_df.columns if col != 'Tarih']
        X_scaled_full = scaler_X.transform(current_df[all_dam_occupancy_columns_base])
        last_sequence_scaled_input = X_scaled_full[-time_steps:].reshape(1, time_steps, X_scaled_full.shape[1])

        future_preds_scaled = []
        for _ in range(forecast_days):
            next_pred_scaled = model.predict(last_sequence_scaled_input, verbose=0)[0,0]
            future_preds_scaled.append(next_pred_scaled)
            
            new_input_features_vector_scaled = last_sequence_scaled_input[0, -1, :].copy() 
            target_dam_index_in_all_dams = all_dam_occupancy_columns_base.index(target_col_name)
            new_input_features_vector_scaled[target_dam_index_in_all_dams] = next_pred_scaled
            
            last_sequence_scaled_input = np.append(last_sequence_scaled_input[:, 1:, :], new_input_features_vector_scaled.reshape(1, 1, -1), axis=1)
        
        forecast_values_inv = scaler_y.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()
        actual_series_for_plot = current_df[['Tarih', target_col_name]]
        
        forecast_lower = None
        forecast_upper = None
        forecast_dates_for_ci = None
        
    # --- Prophet Model Type: Prophet_Multivariate_Extra_Inputs ---
    elif model_type == "Prophet_Multivariate_Extra_Inputs":
        model_path = os.path.join(MODELS_BASE_DIR, dam_name.lower(), "prophet_multivariate_extra_inputs_model.pkl")
        
        current_df = df_merged_specific.copy()
        target_col_name = f"{dam_name}_Fill"
        rainfall_col_for_dam = f"{dam_name}_Rainfall"
        consumption_col_name_base = 'Istanbul_Daily_Consumption'

        model = joblib.load(model_path)

        prophet_cols_for_model = ['Tarih', target_col_name, rainfall_col_for_dam, consumption_col_name_base]
        df_prophet_current_dam = current_df[prophet_cols_for_model].rename(
            columns={'Tarih': 'ds', target_col_name: 'y', rainfall_col_for_dam: 'rainfall', consumption_col_name_base: 'consumption'}
        ).dropna()
        
        future_prophet_dates = model.make_future_dataframe(periods=forecast_days, include_history=False)
        
        synthetic_rainfall_consumption_df_full = df_rainfall_consumption.reset_index().set_index('Tarih')
        
        for col_name_for_prophet_regressor in ['rainfall', 'consumption']:
            if col_name_for_prophet_regressor == 'rainfall':
                original_synthetic_data_col_name = dam_name
            else:
                original_synthetic_data_col_name = consumption_col_name_base
            
            # DÜZELTİLMİŞ KISIM: .fillna() kaldırıldı.
            future_prophet_dates[col_name_for_prophet_regressor] = future_prophet_dates['ds'].map(synthetic_rainfall_consumption_df_full[original_synthetic_data_col_name])
            future_prophet_dates[col_name_for_prophet_regressor].ffill(inplace=True)
            future_prophet_dates[col_name_for_prophet_regressor].bfill(inplace=True)
        
        forecast_prophet = model.predict(future_prophet_dates)
        forecast_values_inv = forecast_prophet['yhat'].values
        
        actual_series_for_plot = df_prophet_current_dam[['ds', 'y']].rename(columns={'ds':'Tarih', 'y':target_col_name})
        
        forecast_lower = forecast_prophet['yhat_lower'].values
        forecast_upper = forecast_prophet['yhat_upper'].values
        forecast_dates_for_ci = forecast_prophet['ds'].values

    # --- General Models ---
    elif model_type.lower() in ["prophet_general_only_occupancy", "sarima_general_only_occupancy"]:
        model_name_in_file = model_type
        if model_type.lower() == "prophet_general_only_occupancy":
             model_name_in_file = "prophet_general_dam_only_occupancy_model" # Dosya adıyla eşleşmesi için
        
        model_path = os.path.join(GENERAL_DAM_MODELS_DIR, f"{model_name_in_file.lower()}.pkl")
        
        current_df = df_occupancy.copy()
        dam_cols_for_general = [col for col in current_df.columns if col != 'Tarih']
        if not dam_cols_for_general:
            st.error("No dam occupancy columns found for general aggregation.")
            return None, None, None, None, None, None, None
        current_df['Aggregated_Dam_Occupancy_Rate'] = current_df[dam_cols_for_general].mean(axis=1)
        target_col_name = 'Aggregated_Dam_Occupancy_Rate'
        
        model = joblib.load(model_path)

        series_for_general_model = current_df.set_index('Tarih')[target_col_name]
        
        # Karşılaştırmada .lower() kullanılıyor
        if model_type.lower() == "prophet_general_only_occupancy":
            df_prophet_general = series_for_general_model.reset_index().rename(columns={'Tarih': 'ds', target_col_name: 'y'})
            future_prophet_general = model.make_future_dataframe(periods=forecast_days, include_history=False)
            forecast_general = model.predict(future_prophet_general)
            forecast_values_inv = forecast_general['yhat'].values
            forecast_lower = forecast_general['yhat_lower'].values
            forecast_upper = forecast_general['yhat_upper'].values
            forecast_dates_for_ci = forecast_general['ds'].values
            actual_series_for_plot = df_prophet_general[['ds', 'y']].rename(columns={'ds':'Tarih', 'y':target_col_name})
            
        elif model_type.lower() == "sarima_general_only_occupancy":
            forecast_sarima_result = model.get_forecast(steps=forecast_days)
            forecast_values_inv = forecast_sarima_result.predicted_mean.values
            forecast_lower = forecast_sarima_result.conf_int(alpha=0.05).iloc[:, 0].values
            forecast_upper = forecast_sarima_result.conf_int(alpha=0.05).iloc[:, 1].values
            forecast_dates_for_ci = forecast_sarima_result.predicted_mean.index.values
            actual_series_for_plot = current_df[['Tarih', target_col_name]] # Tarih'i de alacak şekilde düzeltme
            
        else:
            st.error("Unknown general model type.")
            return None, None, None, None, None, None, None
    
    else:
        st.error(f"Model type {model_type} not recognized or not implemented for forecasting.")
        return None, None, None, None, None, None, None

    # --- Common Forecasting Output ---
    forecast_dates = pd.date_range(actual_series_for_plot['Tarih'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_series = pd.Series(forecast_values_inv, index=forecast_dates)
    
    dam_name_for_metrics = dam_name
    if "Aggregated" in dam_name:
        dam_name_for_metrics = "Aggregated_General_Occupancy"
        
    selected_metrics = df_all_metrics[
        (df_all_metrics['Dam'] == dam_name_for_metrics) &
        (df_all_metrics['Model Type'] == model_type)
    ]
    
    if not selected_metrics.empty:
        mae_val = selected_metrics['MAE'].iloc[0]
        rmse_val = selected_metrics['RMSE'].iloc[0]
        model_metrics = f"MAE: {mae_val:.4f} | RMSE: {rmse_val:.4f}"
    else:
        model_metrics = "Metrics N/A"

    plot_title = f"{model_type} Forecast - {dam_name} Dam" if "Aggregated" not in dam_name else f"{model_type} Forecast - Aggregated General Occupancy"

    return forecast_series, actual_series_for_plot, model_metrics, plot_title, forecast_lower, forecast_upper, forecast_dates_for_ci


# --- Helper function for LSTM sequence creation (moved outside to be global) ---
def create_sequences_univariate_lstm(data, time_steps):
    Xs, ys = [], []
    for i in range(len(data) - time_steps):
        Xs.append(data[i:(i + time_steps)])
        ys.append(data[i + time_steps])
    return np.array(Xs), np.array(ys)


# --- 4. Streamlit UI ---
st.set_page_config(page_title="Istanbul Dams Forecast", layout="wide", initial_sidebar_state="expanded")

st.title("Istanbul Dams Occupancy Forecasting")

# Load all data (cached)
df_occupancy, df_merged_specific, df_rainfall_consumption, df_all_metrics, df_best_per_dam, df_best_2_per_dam = load_all_data()

# Get available dams from the occupancy data
available_dams = [col for col in df_occupancy.columns if col != 'Tarih']

# --- Sidebar Controls ---
st.sidebar.header("Forecast Settings")

# 1. Select Dam or General Occupancy
dam_selection_type = st.sidebar.radio("Select Forecast Scope", ["Individual Dam", "Aggregated General Occupancy"])

selected_dam = None
if dam_selection_type == "Individual Dam":
    selected_dam = st.sidebar.selectbox("Select Dam", available_dams, index=0)
else:
    selected_dam = "Aggregated_General_Occupancy" # Special identifier for general models

# 2. Select Model Strategy
model_strategy = st.sidebar.radio("Select Model Strategy", ["Best Model", "Best 2 Models", "All Models"])

# 3. Dynamic Model Selection based on Strategy
available_models_for_selection = []
if dam_selection_type == "Individual Dam":
    if model_strategy == "Best Model":
        # Find the best model for the selected dam
        best_model_row = df_best_per_dam[df_best_per_dam['Dam'] == selected_dam]
        if not best_model_row.empty:
            available_models_for_selection = [best_model_row['Model Type'].iloc[0]]
        else:
            st.sidebar.warning(f"No best model found for {selected_dam}. Showing all models.")
            available_models_for_selection = df_all_metrics[df_all_metrics['Dam'] == selected_dam]['Model Type'].unique().tolist()
    
    elif model_strategy == "Best 2 Models":
        # Find the best 2 models for the selected dam
        best_2_models_df = df_best_2_per_dam[df_best_2_per_dam['Dam'] == selected_dam]
        if not best_2_models_df.empty:
            available_models_for_selection = best_2_models_df['Model Type'].tolist()
        else:
            st.sidebar.warning(f"No top 2 models found for {selected_dam}. Showing all models.")
            available_models_for_selection = df_all_metrics[df_all_metrics['Dam'] == selected_dam]['Model Type'].unique().tolist()

    elif model_strategy == "All Models":
        available_models_for_selection = df_all_metrics[df_all_metrics['Dam'] == selected_dam]['Model Type'].unique().tolist()

elif dam_selection_type == "Aggregated General Occupancy":
    available_models_for_selection = df_all_metrics[df_all_metrics['Dam'] == "Aggregated_General_Occupancy"]['Model Type'].unique().tolist()


# Ensure there are models to select
if not available_models_for_selection:
    st.error(f"No models found for {selected_dam} with the selected strategy. Please ensure models are trained and metrics are generated.")
    st.stop()

# Select the specific model from the filtered list
selected_model_type = st.sidebar.selectbox("Select Specific Model", available_models_for_selection)

# 4. Forecast Horizon
forecast_horizon_days = st.sidebar.slider("Forecast Horizon (days)", 7, 365, 365)

# --- Analysis Button ---
show_analysis = st.sidebar.button("Analysis")

if st.sidebar.button("Data Analysis"):
    data_analysis_dir = os.path.join(RESULTS_DIR, "plots", "Data Analysis")
    if os.path.exists(data_analysis_dir):
        data_analysis_files = [f for f in os.listdir(data_analysis_dir) if f.endswith('.png')]
        if data_analysis_files:
            st.subheader("Data Analysis Görselleri")
            for data_analysis_file in sorted(data_analysis_files):
                data_analysis_path = os.path.join(data_analysis_dir, data_analysis_file)
                st.image(data_analysis_path, use_column_width=True)
        else:
            st.warning("'Data Analysis' klasöründe png formatında görsel bulunamadı.")
    else:
        st.warning("'Data Analysis' analiz klasörü bulunamadı.")
        
# --- Main Content Area ---
if show_analysis:
    st.header(f"Analysis Plots for {selected_dam}")
    # Construct the path to the dam's plot directory
    # Normalize dam name for path (e.g., 'Omerli' -> 'omerli', 'Omerli_Fill' -> 'omerli')
    dam_folder_name = selected_dam.lower().replace('_fill', '')
    plots_dir = os.path.join(RESULTS_DIR, "plots", dam_folder_name)

    if os.path.exists(plots_dir) and os.path.isdir(plots_dir):
        image_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
        image_files.sort() # Sort to ensure consistent order

        if image_files:
            for img_file in image_files:
                img_path = os.path.join(plots_dir, img_file)
                st.image(img_path, caption=img_file.replace('.png', '').replace('_', ' ').title(), use_column_width=True)
                st.markdown("---") # Add a separator
        else:
            st.warning(f"No PNG images found in the analysis directory for {selected_dam}.")
    else:
        st.error(f"Analysis directory not found for {selected_dam} at: {plots_dir}")

else:
    st.header(f"Forecast for {selected_dam}")
    st.subheader(f"Using: {selected_model_type}")

    if selected_dam is None:
        st.warning("Please select a dam to view the forecast.")
    else:
        # Perform forecasting
        with st.spinner("Generating Forecast..."):
            # Pass df_occupancy, df_merged_specific, df_rainfall_consumption as arguments
            forecast_series, actual_series_for_plot, model_metrics, plot_title, forecast_lower, forecast_upper, forecast_dates_for_ci = \
                load_and_forecast_model(selected_dam, selected_model_type, forecast_horizon_days, df_occupancy, df_merged_specific, df_rainfall_consumption)

            if forecast_series is not None:
                # Plotting the forecast
                fig, ax = plt.subplots(figsize=(18, 7))

                # Actual & Synthetic Data
                # actual_series_for_plot is already a DataFrame with 'Tarih' and target column
                ax.plot(actual_series_for_plot['Tarih'], actual_series_for_plot[actual_series_for_plot.columns[1]], color='darkblue', linewidth=2, label='Actual & Synthetic Data')
                
                # Forecast (from forecast_series)
                ax.plot(forecast_series.index, forecast_series.values, color='crimson', linestyle='-', linewidth=2, label=f'{forecast_horizon_days}-Day Future Forecast')
                
                # Confidence Interval (if available)
                if forecast_lower is not None and forecast_upper is not None:
                    # Ensure forecast_dates_for_ci matches the length of forecast_lower/upper
                    if len(forecast_dates_for_ci) == len(forecast_lower):
                        ax.fill_between(forecast_dates_for_ci, forecast_lower, forecast_upper,
                                        color='lightcoral', alpha=0.3, label='Confidence Interval')
                    else:
                        st.warning("Confidence interval dates mismatch. Plotting without CI.")

                # Original Data End
                original_data_end_date = pd.to_datetime('2024-02-19')
                ax.axvline(original_data_end_date, color='gray', linestyle=':', linewidth=1.5, label='Original Data End')

                ax.set_title(plot_title, fontsize=14)
                ax.set_xlabel("Date")
                ax.set_ylabel("Occupancy Rate (%)")
                ax.legend()
                ax.grid(True)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

                st.subheader("Model Performance")
                st.write(model_metrics)

            else:
                st.warning("Could not generate forecast. Please check model availability or data.")