Istanbul Dam Occupancy Forecasting Project
Project Description
This project focuses on the accurate and reliable forecasting of dam occupancy rates, a critical component of water resource management in a large metropolis like Istanbul. It aims to comparatively analyze various machine learning and statistical modeling approaches using multivariate time series data, including historical dam occupancy rates, rainfall data, and daily water consumption. The ultimate output of this project is an interactive web application built with Streamlit, allowing users to dynamically visualize future forecasts for their selected dam and model.

Features
Comprehensive Data Preprocessing: Handling missing values, outlier management, and data scaling.

Synthetic Data Generation: Extending existing data (ending February 2024) with synthetic occupancy, rainfall, and consumption data up to July 2025.

Prophet model for consumption and occupancy rates.

Hybrid approach based on monthly seasonal statistical sampling for rainfall data.

Diverse Time Series Models:

Univariate LSTM (Occupancy Only): Models using only a dam's own historical occupancy rate.

Multivariate LSTM (Occupancy Only): Models using historical occupancy rates of all dams as input for each individual dam's prediction.

Single Dam Multivariate LSTM (with Extra Inputs): Models incorporating a dam's own occupancy, relevant rainfall, general water consumption, lagged values, and rolling averages.

General Aggregation Models (Prophet & SARIMA): Statistical models forecasting the average occupancy rate across all dams.

All Dams Multivariate LSTM: A system-wide model simultaneously forecasting all dams' occupancy rates, along with their respective rainfall and consumption.

Model Evaluation & Comparison: Quantitative assessment using MAE and RMSE metrics, complemented by qualitative visual inspection of plots.

Overfit Control: Implementation of EarlyStopping and Dropout techniques for LSTM models.

Interactive Web Application: A Streamlit-based interface for dynamic dam and model selection, and forecast visualization.

Setup
To run the project locally, follow these steps:

Clone the Repository:

git clone https://github.com/mertaltunn/dam_forecasting_project.git
cd dam_forecasting_project

Create and Activate a Virtual Environment (Recommended):

python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

Install Required Libraries:

pip install -r requirements.txt


Prepare Data Files:

Place istanbul-dams-daily-occupancy-rates.xlsx and istanbul-barajlarnda-yagis-ve-gunluk-tuketim-verileri.xlsx files into the data/raw/ folder.

Create the data/processed/ folder.

Run Data Preprocessing and Synthetic Data Generation:

Execute the data preprocessing and synthetic data generation scripts (e.g., Jupyter Notebook cells) in your project sequentially to create the necessary cleaned and extended CSV files in the data/processed/ folder (istanbul-dams-daily-occupancy-rates-cleaned_with_synthetic.csv, istanbul-barajlarnda-yagis-ve-gunluk-tuketim-verileri_with_synthetic.csv, merged_dam_specific_extended.csv, merged_general_dam_extended.csv).

Train and Save Models:

Run your model training scripts (Jupyter Notebook cells) to train all model types and save them under the models/ folder according to the specified structure.

Run the model comparison script to generate all_model_metrics.csv, best_model_per_dam.csv, and best_2_models_per_dam.csv files in the results/model_selection/ folder.

Usage
To launch the web application:

streamlit run web_app/app_streamlit.py

The application will open in your browser. You can adjust the settings from the sidebar:

Forecast Scope: Choose whether to forecast for an individual dam or the aggregated general dam occupancy.

Select Dam: If "Individual Dam" is selected, choose a specific dam from the list.

Select Model Strategy:

Best Model: Automatically selects the best-performing model for the chosen dam.

Best 2 Models: Lists the top two best-performing models for the chosen dam, allowing you to select one.

All Models: Lists all available model types for the chosen dam, allowing you to select any.

Select Specific Model: Choose a specific model from the filtered list based on your strategy.

Forecast Horizon (days): Set the number of days for the future forecast (between 7 and 365 days).

[Dam Name] Analysis / Data Analysis Buttons: Used to visualize plots related to the selected dam or general data analysis.

Project Structure
.
├── data/
│   ├── raw/                      # Raw data files
│   │   ├── istanbul-dams-daily-occupancy-rates.xlsx
│   │   └── istanbul-barajlarnda-yagis-ve-gunluk-tuketim-verileri.xlsx
│   └── processed/                # Cleaned and synthetic data files
│       ├── istanbul-dams-daily-occupancy-rates-cleaned_with_synthetic.csv
│       ├── istanbul-barajlarnda-yagis-ve-gunluk-tuketim-verileri_with_synthetic.csv
│       ├── merged_dam_specific_extended.csv
│       └── merged_general_dam_extended.csv
├── models/                       # Trained models and scalers
│   ├── omerli/                   # Specific folders for each dam
│   │   ├── lstm_univariate_only_occupancy_model.h5
│   │   ├── scaler_univariate_only_occupancy.pkl
│   │   ├── lstm_multivariate_extra_inputs_model.h5
│   │   ├── scaler_X_multivariate_extra_inputs.pkl
│   │   └── scaler_y_multivariate_extra_inputs.pkl
│   ├── darlik/                   # ...other dams...
│   ├── general_dam/              # General/aggregation models
│   │   ├── prophet_general_dam_only_occupancy_model.pkl
│   │   ├── sarima_general_dam_only_occupancy_model.pkl
│   │   ├── lstm_general_dam_only_occupancy_model.h5
│   │   ├── scaler_X_general_dam_only_occupancy.pkl
│   │   └── scaler_y_general_dam_only_occupancy.pkl
│   └── all_dams_system_wide/     # System-wide models covering all dams
│       ├── lstm_all_dams_multivariate_model.h5
│       └── scaler_X_all_dams_multivariate.pkl
├── results/
│   ├── plots/                    # Generated plots
│   │   ├── omerli/               # Plot folder for each dam (if generated)
│   │   └── data_analysis/        # Data analysis plots
│   └── model_selection/          # Model comparison results
│       ├── all_model_metrics.csv
│       ├── best_model_per_dam.csv
│       └── best_2_models_per_dam.csv
├── notebooks/                    # Jupyter Notebooks (data prep, model training, comparison)
│   ├── data_preprocessing.ipynb
│   ├── model_training_univariate_lstm.ipynb
│   ├── model_training_multivariate_lstm_occupancy_only.ipynb
│   ├── model_training_single_dam_multivariate_lstm_extra_inputs.ipynb
│   ├── model_training_all_dams_multivariate_lstm.ipynb
│   ├── model_training_prophet_general.ipynb
│   ├── model_training_sarima_general.ipynb
│   └── model_comparison.ipynb
└── web_app/                      # Streamlit application
    └── app_streamlit.py
└── requirements.txt              # Project dependencies

 
