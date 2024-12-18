import pandas as pd
import numpy as np
import random
np.random.seed(42)
random.seed(42)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import shap
from math import sqrt
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR) # Suppress Optuna logs
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsforecast.models import AutoARIMA
from sktime.forecasting.arima import AutoARIMA
from pmdarima import auto_arima
import matplotlib
matplotlib.use('Agg')  # Switch to a non-interactive backend
import matplotlib.pyplot as plt
import os
import joblib
from components.data import prepare_for_arima_ma
import pprint
import traceback
import plotly.graph_objects as go








# ------------------------- Data Splitting for Machine Learning Models -------------------------------------------------------


def split_data(combined_df, train_start, train_end, val_start, val_end, blind_test_start, blind_test_end, target_column):

    train_data = combined_df[(combined_df['Date'] >= train_start) & (combined_df['Date'] <= train_end)]
    val_data = combined_df[(combined_df['Date'] >= val_start) & (combined_df['Date'] <= val_end)]
    blind_test_data = combined_df[(combined_df['Date'] >= blind_test_start) & (combined_df['Date'] <= blind_test_end)]

    #print(f"\nData Spliting Function")
    #print(f"\nRows in Blind Test Data: {blind_test_data}")
    #print(f"\nTrain data size: {train_data.shape[0]}")  #Train data size: 22
    #print(f"\nValidation data size: {val_data.shape[0]}")  #Validation data size: 5
    #print(f"\nBlind test data size: {blind_test_data.shape[0]}")  #Blind test data size: 4
    #print(f"\nEnd of Data Spliting Function")

    X_train = train_data.drop(columns=[target_column, 'Date', 'Country'])
    y_train = train_data[target_column]
    X_val = val_data.drop(columns=[target_column, 'Date', 'Country'])
    y_val = val_data[target_column]
    X_blind_test = blind_test_data.drop(columns=[target_column, 'Date', 'Country'])
    y_blind_test = blind_test_data[target_column]
    
    return (X_train, y_train), (X_val, y_val), (X_blind_test, y_blind_test)  # Return tuples



# ------------------------- Data Splitting for ARIMA/MA Models -------------------------------------------------------

def split_data_for_arima_ma_by_country(prepared_df, train_start, train_end, blind_test_start, blind_test_end, target_column):

    # Considering the cut of 5 first rows, combine train and validation datasets for the training set
    # train_start="2017-10-01", train_end="2023-03-31"
    # blind_test_start="2023-04-01", blind_test_end ="2024-03-31"

    
    # Filter data for the specified time frames
    train_data = prepared_df[(prepared_df.index >= train_start) & (prepared_df.index <= train_end)]
    blind_test_data = prepared_df[(prepared_df.index >= blind_test_start) & (prepared_df.index <= blind_test_end)]

    y_train = train_data[target_column]
    y_blind_test = blind_test_data[target_column]

    return y_train, y_blind_test


def split_data_for_arima_ma(combined_df, train_start, train_end, blind_test_start, blind_test_end, target_column):

    country_dfs = prepare_for_arima_ma(combined_df, target_column)
    #print("countries", country_dfs.keys())

    arima_data = {}
    for country, country_df in country_dfs.items():
        #print("country", country)
        y_train, y_blind_test = split_data_for_arima_ma_by_country(country_df, train_start, train_end, blind_test_start, blind_test_end, target_column)
        arima_data[country] = {"y_train": y_train, "y_blind_test": y_blind_test}
    #print("res", arima_data)
    return arima_data


# ------------------------ Single Default ML Models Developement on Combined Countries -----------------------------------------------------



def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_blind_test=None, y_blind_test=None, combined_df=None, target_column=None):
    # suppress warnings for LightGBM
    if model.__class__.__name__ == 'LGBMClassifier' or model.__class__.__name__ == 'LGBMRegressor':
        model.set_params(verbose=-1)

    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    #print(f"---SINGLE {model}: {val_preds[:5]}")
    metrics = {
        'validation': {
            #'MAPE%': mean_absolute_percentage_error(y_val, val_preds) * 100,
            'Accuracy%': 100 - mean_absolute_percentage_error(y_val, val_preds) * 100,
            'Bias%': (np.mean(val_preds - y_val) / np.mean(y_val)) * 100
        }
    }

    blind_test_preds = None
    if X_blind_test is not None and y_blind_test is not None:
        blind_test_preds = model.predict(X_blind_test)
        metrics['blind_test'] = {
            #'MAPE%': mean_absolute_percentage_error(y_blind_test, blind_test_preds) * 100,
            'Accuracy%': 100 - mean_absolute_percentage_error(y_blind_test, blind_test_preds) * 100,
            'Bias%': (np.mean(blind_test_preds - y_blind_test) / np.mean(y_blind_test)) * 100
        }

    # Add predictions for all countries if combined_df and target_column are provided
    all_countries_preds = None
    if combined_df is not None and target_column is not None:
        X_all = combined_df.drop(columns=[target_column, 'Date', 'Country'], errors='ignore')
        all_countries_preds = model.predict(X_all)

    return model, {
        'metrics': metrics,
        'validation_predictions': val_preds,
        'validation_actuals': y_val,  
        'blind_test_predictions': blind_test_preds,
        'blind_test_actuals': y_blind_test,  
        'all_countries_predictions': all_countries_preds  # Predictions for the entire combined dataset
    }


# --------------------------- Default ML Models Optimization (Bayesian Method) -------------------------------------------


def tune_model(model_class, X_train, y_train, X_val, y_val, trial_params):
    def objective(trial):
        params = {}
        try:
            # Manually specifying each parameter assignment
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 10)
            params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
            params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1.0)
            params['reg_alpha'] = trial.suggest_int('reg_alpha', 0, 1)
            params['reg_lambda'] = trial.suggest_int('reg_lambda', 0, 1)
            params['min_data_in_bin'] = trial.suggest_int('min_data_in_bin', 8, 10) 
            params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 8, 10)

            # For LightGBM-specific parameter if using LightGBM trials
            if 'num_leaves' in trial_params:
                params['num_leaves'] = trial.suggest_int('num_leaves', 20, 50)
        
            #print("Trial parameters generated:", params)

            # suppress warnings for LightGBM
            #if model_class.__name__ == 'LGBMClassifier' or model_class.__name__ == 'LGBMRegressor':
                #params['verbose'] = -1
        
            # Create and train model with these parameters
            model = model_class(**params, random_state=42)
            model.fit(X_train, y_train)
            val_preds = model.predict(X_val)
            return sqrt(mean_squared_error(y_val, val_preds))
        
        except TypeError as e:
            print(f"Error in parameter setting: {e}")
            raise e  


    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    return study.best_params



# ----------------------------------- Single Retrained ML Models Developement on Combined Countries on Blind Test Set -----------------------------------------


def retrain_and_evaluate(model_class, best_params, X_combined, y_combined, X_blind_test, y_blind_test, combined_df=None, target_column=None):
    
    # suppress warnings for LightGBM
    if model_class.__name__ == 'LGBMClassifier' or model_class.__name__ == 'LGBMRegressor':
        best_params['verbose'] = -1

    model = model_class(**best_params, random_state=42)
    model.fit(X_combined, y_combined)
    test_preds = model.predict(X_blind_test)
    #print(f"SINGLE predictions for model {model_class}: {test_preds[:5]}")
    metrics = {
        'blind_test': {
            #'MAPE%': mean_absolute_percentage_error(y_blind_test, test_preds) * 100,
            'Accuracy%': 100 - mean_absolute_percentage_error(y_blind_test, test_preds) * 100,
            'Bias%': (np.mean(test_preds - y_blind_test) / np.mean(y_blind_test)) * 100
        }
    }

    # Predict for all countries if combined_df and target_column are provided
    all_countries_preds = None
    if combined_df is not None and target_column is not None:
        X_all = combined_df.drop(columns=[target_column, 'Date', 'Country'], errors='ignore')
        all_countries_preds = model.predict(X_all)

    return model, {
        'metrics': metrics,  
        'test_preds': test_preds,
        'test_actuals': y_blind_test,  
        'all_countries_predictions': all_countries_preds  # Predictions for the entire combined dataset
    } 

    #model, metrics, test_preds, all_countries_preds



# ----------------------------------- Developing Time Series Model -----------------------------------------

def train_arima_model(y_combined, blind_test_data, max_p=3, max_q=3, max_d=1):
    """
    Train an ARIMA model on the provided target series and predict for the blind test period.

    Args:
        y_combined (pd.Series): Combined target time series (train + validation) for ARIMA training.
        blind_test_data (pd.Series): Target series for the blind test period.
        max_p (int): Maximum order for the AR term.
        max_q (int): Maximum order for the MA term.
        max_d (int): Maximum order for differencing.

    Returns:
        dict: Contains the fitted ARIMA model and predictions for the blind test period.
    """
    try:
        # Validate input series
        if y_combined.isnull().any():
            raise ValueError("y_combined contains NaN values. Please ensure it's cleaned before training the ARIMA model.")
        if len(y_combined) < max_p + max_d + max_q + 1:
            raise ValueError("Insufficient data points for ARIMA model fitting.")

        # Check for stationarity and apply differencing if needed
        if adfuller(y_combined)[1] > 0.05:  # p-value > 0.05 indicates non-stationarity
            y_combined = y_combined.diff().dropna()

        # Automatically determine the best ARIMA parameters
        arima_params_model = auto_arima(
            y_combined, seasonal=False, stepwise=True, trace=False,
            error_action='ignore', max_p=max_p, max_q=max_q, max_d=max_d
        )
        best_params = arima_params_model.order

        # Train the ARIMA model
        arima_model = ARIMA(y_combined, order=best_params)
        arima_model_fitted = arima_model.fit(method_kwargs={"maxiter": 1000})

        # Check for convergence issues
        if not arima_model_fitted.mle_retvals.get('converged', True):
            print(f"Warning: ARIMA model did not converge. Details: {arima_model_fitted.mle_retvals}")

        # Predict for the blind test period
        arima_forecast = arima_model_fitted.forecast(steps=len(blind_test_data)) #.predicted_mean

        # Handle invalid forecasts (NaNs or infs) by forward/backward filling
        arima_forecast = arima_forecast.fillna(method='ffill').fillna(method='bfill')
        #print("TESTING ARIMA", arima_forecast)

        # Return results
        return {
            "arima_model": arima_model,
            "arima_model_fitted": arima_model_fitted,
            "blind_test_predictions": arima_forecast.tolist(),
            "actual_values": blind_test_data
        }

    except Exception as e:
        print(f"Error: ARIMA model training or forecasting failed: {e}")
        return None




def train_ma_model(y_combined, blind_test_data, window):
    """
    Train a simple Moving Average (MA) model and forecast blind test values for each country.

    Args:
        y_combined (pd.Series): Combined target time series (train + validation) for MA training.
        blind_test_data (pd.Series): Target series for the blind test set.
        window (int): Rolling window size for the moving average.

    Returns:
        dict: Contains the rolling mean (MA model) and predictions for the blind test set.
    """
    try:
        # Validate inputs
        if y_combined.isnull().any():
            raise ValueError("y_combined contains NaN values. Ensure the input series is cleaned.")
        if len(y_combined) < window:
            raise ValueError(f"Not enough data points to calculate moving average with window={window}.")
        if y_combined.nunique() <= 1:
            raise ValueError("The input series has no variability.")

        # Calculate the rolling mean
        ma_model = y_combined.rolling(window=window).mean()

        # Extract predictions for the blind test period
        ma_forecast = ma_model.iloc[-len(blind_test_data):]

        #print("TESTING MA", ma_forecast)

        return {
            "ma_model": ma_model,   
            "ma_predictions": ma_forecast.tolist(),
            "actual_values": blind_test_data
        }

    except Exception as e:
        print(f"Error in MA model training or forecasting: {e}")
        return None

# #If the mean of y_blind_test_country (the actual values) is 0, this calculation will result in a division by zero, leading to NaN

# def train_arima_model(y_combined, blind_test_data, blind_test_steps=None, future_forecast_steps = 5, max_p=3, max_q=3, max_d=1):
#     """
#     Train an ARIMA model on the provided target series and forecast future values.

#     Args:
#         y_combined (pd.Series): Combined target time series (train + validation) for ARIMA training.
#         blind_test_data (pd.Series): Target series for the blind test set.
#         forecast_steps (int): Number of future steps to forecast.
#         max_p (int): Maximum order for the AR term.
#         max_q (int): Maximum order for the MA term.
#         max_d (int): Maximum order for differencing.

#     Returns:
#         tuple: (ARIMA model, blind test forecast array, future forecast array)
#     """

#     #print(f"Start of Training ARIMA Model")
#     #print(f"After dropping 5 rows, y_combined length: {len(y_combined)}, blind_test_data length: {len(blind_test_data)}, blind_test_steps: {blind_test_steps}, future_forecast_steps: {future_forecast_steps}")  #y_combined length: 26 (31 total rows - 5 dropped rows), blind_test_data length: 4, forecast_steps: 4
#     #print(f"First few values of y_combined:\n{y_combined.head()}")   #correct values
#     #print(f"Rows in Blind Test Data: {blind_test_data}")
#     #print(f"Rows in Blind Test Steps: {blind_test_steps}")

#     # Default blind test steps to the length of the blind test data
#     if blind_test_steps is None:
#         blind_test_steps = len(blind_test_data)
    
#     # Ensure y_combined has no NaN values
#     if y_combined.isnull().any():
#         print("Warning: NaN values detected in y_combined. Please ensure it's cleaned before passing to ARIMA.")
#         return None, np.full(blind_test_steps, np.nan), np.full(future_forecast_steps, np.nan)

#     # Perform stationarity check
#     try:
#         result = adfuller(y_combined)
#         #print(f"ADF Statistic: {result[0]}")
#         #print(f"p-value: {result[1]}")
#         #print(f"Initial y_combined length before differencing: {len(y_combined)}")
#         if result[1] > 0.05:
#             #print("Non-stationary y_combined (p-value > 0.05). Differencing the data.")
#             y_combined = y_combined.diff().dropna()
#             #print(f"y_combined length after differencing: {len(y_combined)}")
#     except Exception as e:
#         print(f"ADF test failed: {e}")
#         return None, np.full(blind_test_steps, np.nan), np.full(future_forecast_steps, np.nan)

#     # Ensure sufficient data points for ARIMA model fitting
#     if len(y_combined) < max_p + max_d + max_q + 1:
#         print("Warning: Insufficient data points for ARIMA model fitting.")
#         return None, np.full(blind_test_steps, np.nan), np.full(future_forecast_steps, np.nan)

#     try:
#         # Automatically determine best ARIMA parameters
#         arima_params_model = auto_arima(
#             y_combined, seasonal=False, stepwise=True, trace=False,
#             error_action='ignore', max_p=max_p, max_q=max_q, max_d=max_d
#         )
#         best_params = arima_params_model.order
#         #print(f"Best ARIMA order: {best_params}")

#         if not isinstance(y_combined.index, (pd.DatetimeIndex, pd.RangeIndex)):
#             y_combined = y_combined.reset_index(drop=True)
#             print('y_combined index not right type')
#         else:
#             print('y_combined index has right type')

#         # Train ARIMA model with the determined parameters
#         arima_model = ARIMA(y_combined, order=best_params)
#         arima_model_fitted = arima_model.fit(method_kwargs={"maxiter": 1000})

#         # print('type', type(arima_model))
#         # print('type fitted', type(arima_model_fitted))

#         # # -------------
#         # order = arima_model_fitted.model.order  # Extract ARIMA order
#         # data = arima_model_fitted.model.endog   # Extract the original data

#         # # Recreate the model
#         # arima_model_fitted = ARIMA(data, order=order)
#         # print("arima_model_fitted", arima_model_fitted)
#         #----------------

#         # Check for convergence issues and log them
#         if not arima_model_fitted.mle_retvals.get('converged', True):
#             print(f"Convergence warning details: {arima_model_fitted.mle_retvals}")

#         # Forecast for the blind test period
#         arima_forecast = arima_model_fitted.get_forecast(steps=blind_test_steps).predicted_mean  # Used get_forecast to independently calculate arima_forecast and future_arima_forecast
#         #print("Index of arima_forecast:", arima_forecast.index)

#         # Forecast for the future
#         future_arima_forecast = arima_model_fitted.get_forecast(steps=future_forecast_steps).predicted_mean

        
#         # # Handle invalid forecasts with mean method
#         # if arima_forecast.isnull().any() or np.isinf(arima_forecast).any():
#         #     print("Warning: Invalid values in ARIMA blind test forecast. Filling with mean of y_combined.")
#         #     arima_forecast = pd.Series([y_combined.mean()] * blind_test_steps)

#         if future_arima_forecast.isnull().any() or np.isinf(future_arima_forecast).any():
#             print("Warning: Invalid values in ARIMA future forecast. Filling with mean of y_combined.")
#             future_arima_forecast = pd.Series([y_combined.mean()] * future_forecast_steps)
        
#         # Handle invalid forecasts with forward/backward filling method
#         if arima_forecast.isnull().any() or np.isinf(arima_forecast).any():
#             print("Warning: Invalid values in ARIMA blind test forecast. Filling using forward and backward fill.")
#             arima_forecast = arima_forecast.fillna(method='ffill').fillna(method='bfill')

#         # if future_arima_forecast.isnull().any() or np.isinf(future_arima_forecast).any():
#         #     print("Warning: Invalid values in ARIMA future forecast. Filling using forward and backward fill.")
#         #     future_arima_forecast = future_arima_forecast.fillna(method='ffill').fillna(method='bfill')

#         #print(f"ARIMA forecast (blind test): {arima_forecast}")
#         #print(f"ARIMA forecast (future): {future_arima_forecast}")

#         return arima_model_fitted, arima_forecast, future_arima_forecast

#     except Exception as e:
#         print(f"Warning: ARIMA model fitting or forecasting failed: {e}")
#         return None, np.full(blind_test_steps, np.nan), np.full(future_forecast_steps, np.nan)  # Return NaN forecast if fitting fails
    
    


# def train_ma_model(y_combined, blind_test_data, window, forecast_steps):
#     """
#     Train a simple Moving Average (MA) model on the provided target series and forecast future values.

#     Args:
#         y_combined (pd.Series): Combined target time series (train + validation) for MA training.
#         blind_test_data (pd.Series): Target series for the blind test set.
#         window (int): Rolling window size for the moving average.
#         forecast_steps (int): Number of future steps to forecast.

#     Returns:
#         tuple: (Moving Average series, blind test forecast array, future forecast array)
#     """
#     #Ensure y_combined is valid
#     # if y_combined.isnull().any():
#     #     print("Warning: NaN values detected in y_combined. Please ensure it's cleaned before passing to MA model.")
#     #     return None, np.full(len(blind_test_data), np.nan), np.full(forecast_steps, np.nan)

#     # if len(y_combined) < window:
#     #     print(f"Warning: Not enough data points to calculate moving average with window={window}.")
#     #     return None, np.full(len(blind_test_data), np.nan), np.full(forecast_steps, np.nan)

#     # if y_combined.nunique() <= 1:
#     #     print("Warning: The input series has no variability.")
#     #     return None, np.full(len(blind_test_data), np.nan), np.full(forecast_steps, np.nan)
#     print("blind_test_data?", blind_test_data)
#     print("y_combined?", y_combined)
#     try:
#         # Calculate the rolling mean
#         ma_model = y_combined.rolling(window=window).mean() #.shift(1)
#         print("WHAT IS INSIDE MODEL?", ma_model)
#         # Ensure there's enough data in the rolling mean. Forecast for blind test: Use the last known rolling mean
#         # if len(ma_model.dropna()) < 1:
#         #     print("Warning: Not enough non-NaN values in the rolling mean.")
#         #     return None, np.full(len(blind_test_data), np.nan), np.full(forecast_steps, np.nan)

#         # Use the last rolling mean for blind test and future forecast
#         ma_forecast = ma_model.dropna().iloc[-len(blind_test_data):]

#         print("WHAT IS INSIDE FORECAST?", ma_forecast)
#         #print("Last known rolling mean value:", last_known_mean)
#         #ma_forecast = np.full(len(blind_test_data), last_known_mean)  # Extend with last rolling mean
#         #future_ma_forecast = np.full(forecast_steps, last_known_mean)

#         # last_known_mean = ma_model.dropna().iloc[-1]
#         # print("Last known rolling mean value:", last_known_mean)
#         # ma_forecast = np.full(len(blind_test_data), last_known_mean)  # Extend with last rolling mean
#         #future_ma_forecast = np.full(forecast_steps)  # Extend future steps with the same mean
        
#         return ma_model, ma_forecast #, future_ma_forecast
#     except Exception as e:
#         print(f"Warning: MA model calculation or forecasting failed: {e}")
#         return None, np.full(len(blind_test_data), np.nan), np.full(forecast_steps, np.nan)





# ----------------------------------- Model Evaluation on Each Country -----------------------------------------


def evaluate_models_by_country(models, retrained_models, combined_df, X_combined, y_combined, target_column, val_start, val_end, blind_test_start, blind_test_end):
    countries = combined_df['Country'].unique()
    results = {}
    #arima_models = {}
    #ma_models = {}
    #print(f"@###@Initial ARIMA models dictionary: {arima_models}")

    for country in countries:
        #print(f"\n----- Evaluating ML models for country: {country}")
        country_data = combined_df[combined_df['Country'] == country]
        
        
        
        #print(f"arima_df after dropping the first 5 rows for ARIMA:\n{arima_df.head()}")

        # Splitting data for the specific country
        (X_train_country, y_train_country), (X_val_country, y_val_country), (X_blind_test_country, y_blind_test_country) = split_data(
            country_data, train_start="2016-07-01", train_end="2021-12-31", 
            val_start=val_start, val_end=val_end, 
            blind_test_start=blind_test_start, blind_test_end=blind_test_end, 
            target_column=target_column
        )

        #print(f"Before witout dropping rows: y_train size: {len(y_train_country)}, y_val size: {len(y_val_country)}, y_blind_test size: {len(y_blind_test_country)}")
        

        # y_blind_test_country_arima = arima_df[y_blind_test_country].iloc[5:]
        # print(f"y_blind_test_country_arima after dropping the first 5 rows for ARIMA:\n{y_blind_test_country_arima.head()}")


        country_results = {}

        # Evaluate Default ML Models
        #print(f"investigate what I have in models: {models}")
        for model_name, model in models.items():

            #print(f"*** Processing default model: {model_name}")
            try:

                                 

                #y_train_country = y_train_country.iloc[5:]
                #print(f"After dropping rows: y_train size: {len(y_train_country)}, y_val size: {len(y_val_country)}, y_blind_test size: {len(y_blind_test_country)}")
                #if model_name == 'ARIMA':
                
                #print(f"y_combined_arima size for ARIMA training: {len(y_combined_arima)}")

                #print(f"Predicting for validation set with default {model_name}...")
                val_preds = model.predict(X_val_country)
                #print(f"Validation predictions default model {model_name}: {val_preds[:5]} (showing first 5)")

                #print(f"Predicting for blind test set with default {model_name}...")
                blind_test_preds = model.predict(X_blind_test_country)
                #print(f"Blind test predictions default model {model_name}: {blind_test_preds[:5]} (showing first 4)")

                #print(f"Calculating metrics for default {model_name}...")
                metrics = {
                    'validation': {
                        #'MAPE%': mean_absolute_percentage_error(y_val_country, val_preds) * 100,
                        'Accuracy%': 100 - mean_absolute_percentage_error(y_val_country, val_preds) * 100,
                        'Bias%': (np.mean(val_preds - y_val_country) / np.mean(y_val_country)) * 100
                    },
                    'blind_test': {
                        #'MAPE%': mean_absolute_percentage_error(y_blind_test_country, blind_test_preds) * 100,
                        'Accuracy%': 100 - mean_absolute_percentage_error(y_blind_test_country, blind_test_preds) * 100,
                        'Bias%': (np.mean(blind_test_preds - y_blind_test_country) / np.mean(y_blind_test_country)) * 100
                    }
                }
                #print(f"Metrics for default {model_name}: {metrics}")

                #print(f"Storing results for default {model_name}")
                country_results[model_name] = {

                    'metrics': metrics,
                    'validation_predictions': val_preds,
                    'validation_actuals': y_val_country.reset_index(drop=True),
                    'blind_test_predictions': blind_test_preds,
                    'blind_test_actuals': y_blind_test_country.reset_index(drop=True) 
                }
                #print(f"Updated country_results after processing {model_name}:")
                #print(country_results)

            except Exception as e:
                print(f"Error evaluating default {model_name} for {country}: {e}")


        # Evaluate Retrained ML Models
        #print(f"investigate what I have in re-models: {retrained_models}")

        for model_name, model in retrained_models.items():
            #print(f"+++ Processing retrained model: {model_name}")
            try:
                #print(f"Predicting blind test set with retrained model {model_name}")
                blind_test_preds = model.predict(X_blind_test_country)
                #print(f"Blind test predictions: {blind_test_preds[:5]}(showing first 4)")

                #print(f"Calculating metrics for retrained model {model_name}")
                metrics = {
                    'blind_test': {
                        #'MAPE%': mean_absolute_percentage_error(y_blind_test_country, blind_test_preds) * 100,
                        'Accuracy%': 100 - mean_absolute_percentage_error(y_blind_test_country, blind_test_preds) * 100,
                        'Bias%': (np.mean(blind_test_preds - y_blind_test_country) / np.mean(y_blind_test_country)) * 100
                    }
                }
                #print(f"Metrics for retrained model {model_name}: {metrics}") 

                #print(f"Storing results for retrained model {model_name}")
                country_results[model_name] = {
                    'metrics': metrics,
                    'blind_test_predictions': blind_test_preds,
                    'blind_test_actuals': y_blind_test_country.reset_index(drop=True)
                }

                #print(f"Updated country_results after processing retrained model {model_name}:")
                #print(country_results)

            except Exception as e:
                print(f"Error evaluating retrained model {model_name} for {country}: {e}")

            # Pass the combined train and validation target series to ARIMA (why it is reported 2 times in the terminal????)
            #print(f"Final y_combined length before ARIMA for country: {country}: {len(y_combined)}")  #Final y_combined length before ARIMA for country: Spain: 27
            #print(f"Final y_combined indices before ARIMA for country: {country}:\n{y_combined.index}")   #Final y_combined indices before ARIMA for country: Spain: RangeIndex(start=0, stop=27, step=1)
            #print(f"Final y_combined head before ARIMA for country: {country}:\n{y_combined.head()}")   # correct values
        
        # # Evaluate ARIMA Model
        # try:

        #     # Use arima_df for ARIMA and MA models
        #     arima_df = prepare_for_arima_ma(country_data, target_column)

        #     #print('arima columns', arima_df.columns)
        #     #print('combined df columns', combined_df.columns)
        #     arima_df.reset_index(inplace=True)


        #     (X_train_country, y_train_country), (X_val_country, y_val_country), (X_blind_test_country, y_blind_test_country) = split_data(
        #     arima_df, train_start="2017-10-01", train_end="2021-12-31", 
        #     val_start=val_start, val_end=val_end, 
        #     blind_test_start=blind_test_start, blind_test_end=blind_test_end, 
        #     target_column=target_column
        #     )
        #     #print('confirm data pass', combined_df.head())

        #     # Combine train and validation for ARIMA
        #     y_combined_arima = pd.concat([y_train_country, y_val_country])

        #     future_forecast_steps = 5
        #     arima_model, arima_forecast, future_arima_forecast = train_arima_model(
        #         y_combined=y_combined_arima,  # Use the full ARIMA-preprocessed target column (arima_df[target_column])
        #         blind_test_data=y_blind_test_country,  # Pass the blind test set
        #         blind_test_steps=len(y_blind_test_country),
        #         future_forecast_steps=future_forecast_steps  # Forecast steps equal to the length of the blind test set
        #     )

        #     #print(f"Blind test forecast length: {len(arima_forecast)}")   #Blind test forecast length: 4
        #     #print(f"Future forecast length: {len(future_arima_forecast)}")  #Future forecast length: 8
        #     #print(f"Mean of actual values (y_blind_test_country): {np.mean(y_blind_test_country)}")
        #     #print(f"Mean of forecasted values (arima_forecast): {np.mean(arima_forecast)}")
        #     #print(f"y_blind_test_country:\n{y_blind_test_country}")
        #     #print(f"arima_forecast:\n{arima_forecast}")
        #     #print(f"Mean of y_blind_test_country: {np.mean(y_blind_test_country)}")
        #     #print(f"Mean of arima_forecast: {np.mean(arima_forecast)}")

        #     # Add model to dictionary
        #     arima_models[country] = arima_model
        #     #print(f"@###@Updated ARIMA models dictionary: {arima_models}")

        #     #arima_forecast_reset = arima_forecast.reset_index(drop=True)
        #     #y_blind_test_reset = y_blind_test_country.reset_index(drop=True)
        #     #print(f"Mean of actual values (y_blind_test_country): {np.mean(y_blind_test_country)}")
        #     #print(f"Mean of forecasted values (arima_forecast): {np.mean(arima_forecast)}")
        #     #print(np.mean(arima_forecast - y_blind_test_country))
        #     arima_metrics = {
        #         'blind_test': {
        #             'MAPE%': mean_absolute_percentage_error(y_blind_test_country, arima_forecast) * 100,
        #             'Accuracy%': 100 - mean_absolute_percentage_error(y_blind_test_country, arima_forecast) * 100,
        #             'Bias%': np.nan if np.mean(y_blind_test_country) == 0 else (np.mean(arima_forecast - y_blind_test_country) / np.mean(y_blind_test_country)) * 100  #Avoided division by zero when calculating 
        #         }
        #     }
        #     country_results['ARIMA'] = {
        #         'metrics': arima_metrics,                
        #         'blind_test_predictions': arima_forecast,
        #         'blind_test_actuals': y_blind_test_country,
        #         'future_forecast': future_arima_forecast
        #     }

        #     # Debugging: Verify the contents of the ARIMA results
        #     #print(f"Country ARIMA results: {country_results['ARIMA']}")

        # except Exception as e:
        #     print(f"ARIMA model evaluation failed for {country}: {e}")
        #     traceback.print_exc()  # Print the stack trace

        # # Evaluate Moving Average Model
        # try:

        #     # Use arima_df for ARIMA and MA models
        #     #arima_df = prepare_for_arima_ma(country_data, target_column)
        #     #arima_df.reset_index(inplace=True)
        #     # (X_train_country, y_train_country), (X_val_country, y_val_country), (X_blind_test_country, y_blind_test_country) = split_data(
        #     # country_data, train_start="2016-07-01", train_end="2021-12-31", 
        #     # val_start=val_start, val_end=val_end, 
        #     # blind_test_start=blind_test_start, blind_test_end=blind_test_end, 
        #     # target_column=target_column
        #     # )

            

        #     ma_model, ma_forecast = train_ma_model(
        #         y_combined=y_combined,  # Use the full MA-preprocessed target column (arima_df[target_column])
        #         blind_test_data=y_blind_test_country,  # Pass the blind test set
        #         window=2,  # Rolling window size for MA
        #         forecast_steps=len(y_blind_test_country)  # Forecast steps equal to the length of the blind test set
        #     )

        #     ma_models[country] = ma_model
        #     print("Mean preds-actuals bias", np.mean(ma_forecast - y_blind_test_country))
        #     ma_forecast_reset = ma_forecast.reset_index(drop=True)
        #     y_blind_test_country_reset = y_blind_test_country.reset_index(drop=True)
        #     ma_metrics = {
        #         'blind_test': {
        #             'MAPE%': mean_absolute_percentage_error(y_blind_test_country, ma_forecast) * 100,
        #             'Accuracy%': 100 - mean_absolute_percentage_error(y_blind_test_country, ma_forecast) * 100,
        #             'Bias%': np.nan if np.mean(y_blind_test_country) == 0 else (np.mean(ma_forecast_reset - y_blind_test_country_reset) / np.mean(y_blind_test_country_reset)) * 100 #'Bias%': (np.mean(ma_forecast - y_blind_test_country) / np.mean(y_blind_test_country)) * 100
        #         }
        #     }
        #     print("WHAT IS ACTUALS?", y_blind_test_country)
        #     country_results['Moving Average'] = {
        #         'metrics': ma_metrics,
        #         'blind_test_predictions': ma_forecast,
        #         'blind_test_actuals': y_blind_test_country.reset_index(drop=True),
        #         #'future_forecast': future_ma_forecast
        #     }
        #     #print(f"Country MA results: {country_results['Moving Average']}")

        # except Exception as e:
        #     print(f"Moving Average model evaluation failed for {country}: {e}")

        results[country] = country_results

    #print(f"FINAL MACHINE LEARNING: {results}")
    return results #, arima_models, ma_models




def evaluate_arima_model_by_country(combined_df, target_column):
    """
    Evaluate ARIMA models for each country and calculate metrics.

    Args:
        combined_df (pd.DataFrame): The combined dataset containing all countries.
        target_column (str): The target column for ARIMA model evaluation.
        blind_test_start (str): Start date for the blind test period.
        blind_test_end (str): End date for the blind test period.

    Returns:
        dict: Contains ARIMA model metrics and predictions for each country.
    """
    #print("****START EVALUATING ARIMA")
    #countries = country_dfs[Country]
    arima_results = {}
    arima_data = split_data_for_arima_ma(combined_df
                                         , train_start="2017-10-01"
                                         , train_end="2023-03-31"
                                         , blind_test_start="2023-04-01"
                                         , blind_test_end="2024-03-31"
                                         , target_column=target_column)

    #print("team", arima_data)
    for country, country_data in arima_data.items():
        #print("country data in arima", type(country_data))
        try:
            # Filter data for the current country
            #country_data = country_dfs[country_dfs[Country] == country]

            # Preprocess data for ARIMA
            #arima_df = prepare_for_arima_ma(country_data, target_column)

            # Split the data for training and testing
            y_train_country = country_data["y_train"]
            #print("type y_train", type(y_train_country))
            y_blind_test_country = country_data["y_blind_test"]
            #print("type y_blind_test_country", type(y_blind_test_country))
            #print("y_blind_test_country", y_blind_test_country)
            #print("y_train_country", y_train_country)
           

            # Train the ARIMA model and predict for the blind test period
            arima_model_dict = train_arima_model(
                y_combined=y_train_country,
                blind_test_data=y_blind_test_country
            )
            arima_model = arima_model_dict["arima_model"]
            arima_forecast = arima_model_dict["blind_test_predictions"]
            y_blind_test_country = arima_model_dict["actual_values"]

            #print("arima_forecast", arima_forecast)
            #print("arima_model", arima_model)

            # Calculate metrics for the ARIMA model
            arima_metrics = {
                'blind_test': {
                    #'MAPE%': mean_absolute_percentage_error(y_blind_test_country, arima_forecast) * 100,
                    'Accuracy%': 100 - mean_absolute_percentage_error(y_blind_test_country, arima_forecast) * 100,
                    'Bias%': (np.mean(arima_forecast - y_blind_test_country) / np.mean(y_blind_test_country)) * 100
                }
            }

            #print("ARIMA METRICS", arima_metrics)

            # Store results for the current country
            arima_results[country] = {
                'metrics': arima_metrics,
                'blind_test_predictions': arima_forecast,
                'blind_test_actuals': y_blind_test_country, #.reset_index(drop=True).tolist(),
                'fitted_model': arima_model
            }

            #print("ARIMA FINAL RESULTS", arima_results)

        except Exception as e:
            print(f"Error evaluating ARIMA model for {country}: {e}")

    return arima_results





def evaluate_ma_model_by_country(combined_df, target_column): #, blind_test_start, blind_test_end, window):
    """
    Evaluate Moving Average (MA) models for each country and calculate metrics.

    Args:
        combined_df (pd.DataFrame): The combined dataset containing all countries.
        target_column (str): The target column for MA model evaluation.
        blind_test_start (str): Start date for the blind test period.
        blind_test_end (str): End date for the blind test period.
        window (int): Rolling window size for the MA model.

    Returns:
        dict: Contains MA model metrics and predictions for each country.
    """
    #print("****START EVALUATING MA")
    #countries = country_dfs['Country'].unique()
    ma_results = {}
    ma_data = split_data_for_arima_ma(combined_df
                                         , train_start="2017-10-01"
                                         , train_end="2023-03-31"
                                         , blind_test_start="2023-04-01"
                                         , blind_test_end="2024-03-31"
                                         , target_column = target_column)

    for country, country_data in ma_data.items():
        try:
            y_train_country = country_data["y_train"]
            y_blind_test_country = country_data["y_blind_test"]

        

            # Train the MA model and predict for the blind test period
            ma_model_dict = train_ma_model(
                y_combined=y_train_country,
                blind_test_data=y_blind_test_country,
                window=2
            )

            #ma_model = ma_model_dict["ma_model"]
            ma_forecast = ma_model_dict["ma_predictions"]
            y_blind_test_country = ma_model_dict["actual_values"]

            # Calculate metrics for the MA model
            ma_metrics = {
                'blind_test': {
                    #'MAPE%': mean_absolute_percentage_error(y_blind_test_country, ma_forecast) * 100,
                    'Accuracy%': 100 - mean_absolute_percentage_error(y_blind_test_country, ma_forecast) * 100,
                    'Bias%': (np.mean(ma_forecast - y_blind_test_country) / np.mean(y_blind_test_country)) * 100
                }
            }

            #print("****END EVALUATING MA", ma_metrics)

            # Store results for the current country
            ma_results[country] = {
                'metrics': ma_metrics,
                'blind_test_predictions': ma_forecast,
                'blind_test_actuals': y_blind_test_country #.reset_index(drop=True).tolist()
            }

            #print("MA FINAL RESULTS", ma_results)

        except Exception as e:
            print(f"Error evaluating Moving Average model for {country}: {e}")

    return ma_results





# ---------------------------------- Back Testing ----------------------------------------------------------------

# # model_dicts is models
# # model_types is the dictionary defining each model's type

def run_backtest(model_dicts, model_types, combined_df, target_column, cycles=3, n_lags=4):
    """
    Run backtesting for all models (ML, ARIMA, MA) on the combined dataset.

    Args:
        model_dicts (list): List of model dictionaries (e.g., ML models, retrained models).
        model_types (dict): Dictionary mapping model names to their types (e.g., 'xgb', 'arima', 'ma').
        combined_df (pd.DataFrame): Combined dataset with all countries.
        target_column (str): The target column for predictions.
        cycles (int): Number of backtesting cycles.
        n_lags (int): Number of future time steps (lags) to predict.

    Returns:
        dict: Backtesting results for each country and model.
    """
    # Combine all ML models and types into one dictionary
    all_models = {}
    for model_dict in model_dicts:
        all_models.update(model_dict)

    countries = combined_df['Country'].unique()
    backtesting_results = {}

    arima_results = evaluate_arima_model_by_country(combined_df, target_column)
    arima_models = {country: arima_results[country]["fitted_model"] for country in arima_results.keys()}
    arima_data = split_data_for_arima_ma(combined_df
                                         , train_start="2017-10-01"
                                         , train_end="2023-03-31"
                                         , blind_test_start="2023-04-01"
                                         , blind_test_end="2024-03-31"
                                         , target_column=target_column)
    
    for country in countries:
        #print(f"\nProcessing country in back testing: {country}")
        print("COUNTRY", country)
        country_data = combined_df[combined_df['Country'] == country]

        # Preprocess data for ARIMA/MA
        ########arima_df = prepare_for_arima_ma(country_data, target_column)

        # Drop the first 5 rows of each country's data for ARIMA
        #arima_df = arima_df.iloc[4:]
        #print(f"arima_df after dropping the first 4 rows for ARIMA:\n{arima_df.head()}")

        ############y_train_arima = arima_df[target_column]

        model_results = {}
        # Initialize global list to store predictions from all cycles
        all_preds = {} 
        all_cycle_metrics = {} 
        for model_name, model in all_models.items():
            model_type = model_types[model_name]
            #print(f"Back testing evaluating model: {model_name} (Type: {model_type})") 
            # Perform backtesting with n_lags
            all_cycle_metrics, all_preds = backtest_model(country_data, [], model, model_name, model_type, target_column, all_preds, all_cycle_metrics, cycles=cycles, n_lags=n_lags)
            #model_results[model_name] = all_cycle_metrics
        

        arima_country_data = arima_data[country]
        arima_model = arima_models[country]
        arima_model_type="arima"

        #ma_country_data = arima_data[country]
        #ma_model = arima_models[country]
        #ma_model_type="ma"

        all_cycle_metrics, all_preds = arima_backtest_model_by_country(arima_country_data, arima_model, arima_model_type, target_column, all_preds, all_cycle_metrics, window=22, cycles=3, n_lags=4)
        #print("before ma", all_cycle_metrics, all_preds)
        all_cycle_metrics, all_preds = ma_backtest_model_by_country(arima_country_data, all_preds, all_cycle_metrics, window=22, cycles=3, n_lags=4)
        #print("after ma", all_cycle_metrics, all_preds)
        #print(f"Completed backtesting for {country}.")
        #preparing actuals for specific plot
        a = len(country_data) - n_lags - cycles
        actual_periods = (country_data['Year'][a:].astype(str) + ' Q' + country_data['Quarter'][a:].astype(str)).tolist()
        preds_actuals = country_data.iloc[a:][target_column].tolist()
        print("MEANINGFUL", a, preds_actuals, actual_periods)
        backtesting_results[country] = {'predictions': all_preds, 'metrics': all_cycle_metrics, 'preds_actuals': preds_actuals, 'actual_periods': actual_periods}
        #print(f'metrics and preds inside , {all_preds}')

    #print("Backtesting complete.")
    return backtesting_results






def backtest_model(country_data, y_train_arima, model, model_name, model_type, target_column, all_preds, all_cycle_metrics, window=27, cycles=3, n_lags=4):
    """
    Perform backtesting for a single model (ML, ARIMA, MA) on a country's data.

    Args:
        country_data (pd.DataFrame): Full dataset for the specific country.
        arima_df (pd.DataFrame): Preprocessed dataset for ARIMA/MA models.
        model (object): The model to be backtested (e.g., ML, ARIMA, MA).
        model_type (str): The type of the model ('xgb', 'arima', 'ma').
        target_column (str): The target column for predictions.
        window (int): Training window size.
        cycles (int): Number of backtesting cycles.
        n_lags (int): Number of future time steps (lags) to predict.

    Returns:
        dict: Metrics for each backtesting cycle.
    """
    #print ("country data in ML", country_data)
    #print(f"Starting backtest for model type: {model_type}")  
    #cycle_metrics = []

    # # Initialize global list to store predictions from all cycles
    # all_preds = []  

    # Determine model type
    #is_ml_model = model_type in ['xgb', 'lgb']
    #is_arima_model = model_type == 'arima'
    #is_ma_model = model_type == 'ma'
    # print("is_ma_model", is_ma_model )
    # print("is_arima_model", is_arima_model )
    # print("is_ml_model", is_ml_model )

    
    for cycle in range(cycles):
        #print(f"Cycle {cycle + 1}/{cycles}")  
        train_end = window - cycle
        #print(f"  Calculated train_end index: {train_end}")
        train_data = country_data.iloc[:train_end]
        #print(f"  Train data size: {train_data.shape[0]}")  
        #print(f"  Train data head:\n{train_data.head()}")
        test_data = country_data.iloc[train_end:train_end + n_lags]
        #print(f"  Test data size: {test_data.shape[0]}")  
        #print(f"  Test data head:\n{test_data.head()}")

        b_X_train = train_data.drop([target_column, 'Date', 'Country'], axis=1, errors='ignore')
        b_y_train = train_data[target_column]
        b_X_test = test_data.drop([target_column, 'Date', 'Country'], axis=1, errors='ignore')
        b_y_test = test_data[target_column]

        #print("lenght", len(b_X_train))
        #print("lenght", len(b_y_train))
        #print(f"Evaluating model Type: {model_type}")
        #print(f"is_ml_model: {is_ml_model}, is_ma_model: {is_ma_model}, is_arima_model: {is_arima_model}")

        # Initialize predictions for the current cycle
        preds = []

        

        #if is_ml_model:
            #print(f"Back testing training ML model.")  
        try:
            model.fit(b_X_train, b_y_train)
            preds = model.predict(b_X_test).tolist()
            #print("after prediction", type(preds), preds)
        except Exception as e:
            print(f"ML model prediction failed: {e}")
            #preds = [np.nan] * len(b_X_test)

                #print(f"Evaluating model is_ml_model: {is_ml_model}, is_ma_model: {is_ma_model}, is_arima_model: {is_arima_model}")

        

        # Append predictions for this cycle to the global list
        #print(f"Before appending, preds for cycle {cycle + 1}: {preds}")
        if model_name not in all_preds.keys():
            all_preds[model_name] = {}
        all_preds[model_name][f'cycle {cycle + 1}'] = preds #.extend(preds)  # Use `.extend()` to add multiple predictions to the global list

        print(f"Updated all_preds after Cycle {cycle + 1}: {all_preds}")

        # Calculate metrics for each lag
        metrics = []
        for lag, (y_true, y_pred) in enumerate(zip(b_y_test, preds), start=1):
            bias = (y_pred - y_true) / y_true * 100
            mape = mean_absolute_percentage_error([y_true], [y_pred]) * 100
            accuracy = 100 - mape

            metrics.append({
                'lag': lag,
                'bias': bias,
                'mape': mape,
                'accuracy': accuracy
            })

            #print(f"Lag {lag}: Bias={bias:.2f}%, Accuracy={accuracy:.2f}%, MAPE={mape:.2f}%")
            
        if model_name not in all_cycle_metrics.keys():
            all_cycle_metrics[model_name] = {}

        all_cycle_metrics[model_name][f'cycle {cycle + 1}'] = {
            #'cycle': cycle + 1,
            'metrics': metrics,
            'mean_mape': np.mean([m['mape'] for m in metrics]),  # All values valid, no filtering needed
            'mean_accuracy': np.mean([m['accuracy'] for m in metrics]),  # All values valid, no filtering needed
            'mean_bias': np.mean([m['bias'] for m in metrics if 'bias' in m])
        }

        
        
    #print("Backtesting complete.")
    #print(f"Final accumulated predictions across all cycles: {all_preds}")

    return all_cycle_metrics, all_preds


# def arima_backtest_model(arima_data, models, model_type, target_column, all_preds, all_cycle_metrics, window=22, cycles=3, n_lags=4):
    
#     for country, country_data in arima_data.items():
#         print("country", country)
#         model = models[country]
#         all_cycle_metrics, all_preds = arima_backtest_model_by_country(country_data, model, model_type, target_column, all_preds, all_cycle_metrics, window=22, cycles=3, n_lags=4)
#     return all_cycle_metrics, all_preds


def arima_backtest_model_by_country(country_data, model, model_type, target_column, all_preds, all_cycle_metrics, window=22, cycles=3, n_lags=4):
    
    #print(f"Starting backtest for model type: {model_type}")  
    #cycle_metrics = []

    # # Initialize global list to store predictions from all cycles
    # all_preds = []  

    # Determine model type
    #is_arima_model = model_type == 'arima'
    # print("is_arima_model", is_arima_model )
    model_name = "ARIMA"
    #print("START ARIMA")
    #print("country data", country_data)
    #print("after", country_data)
    combined_series = pd.concat([country_data['y_train'], country_data['y_blind_test']]).reset_index(drop=True)
    #print("new", combined_series)
    for cycle in range(cycles):
        #print(f"Cycle {cycle + 1}/{cycles}")  
        train_end = window - cycle
        #print(f"  Calculated train_end index: {train_end}")
        train_data = combined_series.iloc[:train_end]
        #print(f"  Train data size: {train_data.shape[0]}")  
        #print(f"  Train data head:\n{train_data.head()}")
        test_data = combined_series.iloc[train_end:train_end + n_lags]
        #print(f"  Test data size: {test_data.shape[0]}")  
        #print(f"  Test data head:\n{test_data.head()}")

        # #b_X_train = train_data.drop([target_column, 'Date', 'Country'], axis=1, errors='ignore')
        # b_y_train = train_data[target_column]
        # print("here", b_y_train)
        # #b_X_test = test_data.drop([target_column, 'Date', 'Country'], axis=1, errors='ignore')
        # b_y_test = test_data[target_column]
        # print("what", b_y_test)

        #print(f"Evaluating model Type: {model_type}")
        #print(f"is_ml_model: {is_ml_model}, is_ma_model: {is_ma_model}, is_arima_model: {is_arima_model}")

        # Initialize predictions for the current cycle
        preds = []

        try:
            #for country, country_model in model.items():
                #print(f"ARIMA model type: {type(model)}")

            

            arima_model = auto_arima(
                train_data, seasonal=False, stepwise=True, trace=False,
                error_action='ignore', max_p=2, max_q=2, max_d=1, suppress_warnings=True
        )
            #best_params = arima_params_model.order
            #arima_model = ARIMA(train_data, order=best_params)
            arima_model.fit(train_data)

            # # Check for convergence issues
            # if not arima_model.mle_retvals.get('converged', True):
            #     print(f"Warning: ARIMA model did not converge. Details: {arima_model.mle_retvals}")

            preds = arima_model.predict(n_periods=len(test_data))

            # Handle invalid forecasts (NaNs or infs) by forward/backward filling
            preds = preds.fillna(method='ffill').fillna(method='bfill').tolist()
            
            #preds = forecast.predicted_mean.tolist()
            #print("preds", preds)
                
        except Exception as e:
            print(f"Back testing ARIMA prediction failed: {e}")
        #     #preds.append(np.nan)

           
        # # Ensure predictions align with test data
        # if len(preds) != len(b_y_test):
        #     #print(f"Prediction length mismatch: preds={len(preds)}, b_y_test={len(b_y_test)}")  # Debugging statement
        #     if len(preds) == 0:
        #         # Handle empty predictions by filling with mean of training set
        #         print("Warning: Predictions are empty. Filling with mean of training data.")
        #         train_mean = b_y_train.mean()
        #         preds = list(np.full(len(b_y_test), train_mean if not np.isnan(train_mean) else 0))

        #     elif len(preds) > len(b_y_test):
        #         # Truncate predictions to match test set length
        #         print("Warning: Predictions are longer than test set. Truncating to match length.")
        #         preds = preds[:len(b_y_test)]

        #     elif len(preds) < len(b_y_test):
        #         # Extend predictions with the last value to match test set length
        #         print("Warning: Predictions are shorter than test set. Padding with last prediction value.")
        #         last_valid_pred = preds[-1] if not np.isnan(preds[-1]) else (np.nanmean(preds) if not np.isnan(np.nanmean(preds)) else 0)
        #         padding = [last_valid_pred] * (len(b_y_test) - len(preds))
        #         preds = list(np.append(preds, padding))

        # Append predictions for this cycle to the global list
        #print(f"Before appending, preds for cycle {cycle + 1}: {preds}")
        
        if model_name not in all_preds.keys():
            all_preds[model_name] = {}
        all_preds[model_name][f'cycle {cycle + 1}'] = preds #.extend(preds)  # Use `.extend()` to add multiple predictions to the global list

        #print(f"Updated all_preds after Cycle {cycle + 1}: {all_preds}")

        # Calculate metrics for each lag
        metrics = []
        for lag, (y_true, y_pred) in enumerate(zip(test_data, preds), start=1):
            bias = (y_pred - y_true) / y_true * 100
            #mape = mean_absolute_percentage_error([y_true], [y_pred]) * 100
            accuracy = 100 - (mean_absolute_percentage_error([y_true], [y_pred]) * 100)

            metrics.append({
                'lag': lag,
                'bias': bias,
                #'mape': mape,
                'accuracy': accuracy
            })

            #print(f"Lag {lag}: Bias={bias:.2f}%, Accuracy={accuracy:.2f}%, MAPE={mape:.2f}%")
            
        if model_name not in all_cycle_metrics.keys():
            all_cycle_metrics[model_name] = {}

        all_cycle_metrics[model_name][f'cycle {cycle + 1}'] = {
            #'cycle': cycle + 1,
            'metrics': metrics,
            #'mean_mape': np.mean([m['mape'] for m in metrics]),  # All values valid, no filtering needed
            'mean_accuracy': np.mean([m['accuracy'] for m in metrics]),  # All values valid, no filtering needed
            'mean_bias': np.mean([m['bias'] for m in metrics if 'bias' in m])
        }
    #print("ALL", all_cycle_metrics)
        
        
    #print("Backtesting complete.")
    #print(f"Final accumulated predictions across all cycles: {all_preds}")

    return all_cycle_metrics, all_preds



def ma_backtest_model_by_country(country_data, all_preds, all_cycle_metrics, window=22, cycles=3, n_lags=4):
    
    #print(f"Starting backtest for model type: {model_type}")  
    
    model_name = "ma"
    #print("START MA")
    #print("country data", country_data)
    #print("after", country_data)
    combined_series = pd.concat([country_data['y_train'], country_data['y_blind_test']]).reset_index(drop=True)
    #print("new", combined_series)
    for cycle in range(cycles):
        #print(f"Cycle {cycle + 1}/{cycles}")  
        train_end = window - cycle
        #print(f"  Calculated train_end index: {train_end}")
        train_data = combined_series.iloc[:train_end]
        #print(f"  Train data size: {train_data.shape[0]}")  
        #print(f"  Train data head:\n{train_data.head()}")
        test_data = combined_series.iloc[train_end:train_end + n_lags]
        #print(f"  Test data size: {test_data.shape[0]}")  
        #print(f"  Test data head:\n{test_data.head()}")

        # #b_X_train = train_data.drop([target_column, 'Date', 'Country'], axis=1, errors='ignore')
        # b_y_train = train_data[target_column]
        # print("here", b_y_train)
        # #b_X_test = test_data.drop([target_column, 'Date', 'Country'], axis=1, errors='ignore')
        # b_y_test = test_data[target_column]
        # print("what", b_y_test)

        #print(f"Evaluating model Type: {model_type}")
        #print(f"is_ml_model: {is_ml_model}, is_ma_model: {is_ma_model}, is_arima_model: {is_arima_model}")

        # Initialize predictions for the current cycle
        preds = []

        try:
            # Calculate the rolling mean
            ma_model = train_data.rolling(window=2).mean()

            # Extract predictions for the blind test period
            preds = ma_model.iloc[-len(test_data):].tolist()
            #print("preds", preds)
                
        except Exception as e:
            print(f"Back testing ARIMA prediction failed: {e}")
        #     #preds.append(np.nan)

           
        
        #print(f"Before appending, preds for cycle {cycle + 1}: {preds}")
        
        if model_name not in all_preds.keys():
            all_preds[model_name] = {}
        all_preds[model_name][f'cycle {cycle + 1}'] = preds #.extend(preds)  # Use `.extend()` to add multiple predictions to the global list

        #print(f"Updated all_preds after Cycle {cycle + 1}: {all_preds}")

        # Calculate metrics for each lag
        metrics = []
        for lag, (y_true, y_pred) in enumerate(zip(test_data, preds), start=1):
            bias = (y_pred - y_true) / y_true * 100
            #mape = mean_absolute_percentage_error([y_true], [y_pred]) * 100
            accuracy = 100 - (mean_absolute_percentage_error([y_true], [y_pred]) * 100)

            metrics.append({
                'lag': lag,
                'bias': bias,
                #'mape': mape,
                'accuracy': accuracy
            })

            #print(f"Lag {lag}: Bias={bias:.2f}%, Accuracy={accuracy:.2f}%, MAPE={mape:.2f}%")
            
        if model_name not in all_cycle_metrics.keys():
            all_cycle_metrics[model_name] = {}

        all_cycle_metrics[model_name][f'cycle {cycle + 1}'] = {
            #'cycle': cycle + 1,
            'metrics': metrics,
            #'mean_mape': np.mean([m['mape'] for m in metrics]),  # All values valid, no filtering needed
            'mean_accuracy': np.mean([m['accuracy'] for m in metrics]),  # All values valid, no filtering needed
            'mean_bias': np.mean([m['bias'] for m in metrics if 'bias' in m])
        }
    #print("ALL", all_cycle_metrics)
        
        
    #print("Backtesting complete.")
    #print(f"Final accumulated predictions across all cycles: {all_preds}")

    return all_cycle_metrics, all_preds

# try:
        #     if is_ml_model:
        #         print(f"Back testing training ML model.")  
        #         model.fit(b_X_train, b_y_train)
        #         preds = model.predict(b_X_test).tolist()

        #     if is_arima_model or is_ma_model:
        #         window = 25

        #     elif is_ma_model:
        #         print(f"Back testing training MA model.")  
        #         rolling_mean = train_data[target_column].rolling(window=25).mean()
        #         preds = rolling_mean.iloc[-len(test_data):].tolist()

        #     elif is_arima_model:
        #         print(f"Back testing training ARIMA model.")  
        #         model.fit(train_data[target_column])
        #         forecast = model.forecast(steps=len(test_data))
        #         preds = forecast.tolist()

        #     else:
        #         print("Model type not recognized. Skipping cycle.")
        #         preds = [np.nan] * len(test_data)

        # except Exception as e:
        #     print(f"Model training or prediction failed: {e}")
        #     preds = [np.nan] * len(test_data)


# elif is_ma_model:
        #     print(f"Back testing training MA model.")  
        #     for lag in range(1, n_lags + 1):
        #         try:
        #             #print(f"MA model type: {type(model)}")

        #             forecast = train_data[target_column].rolling(window=2).mean().iloc[-lag]
        #             preds.append(forecast)
        #         except Exception as e:
        #             print(f"Moving Average prediction failed: {e}")
        #             preds.append(np.nan)

        # elif is_arima_model:
        #     print(f"Back testing training ARIMA model.")  
        #     for lag in range(1, n_lags + 1):
        #         try:
        #             print(f"ARIMA model type: {type(model)}")

        #             forecast = model.forecast(steps=lag)
        #             preds.append(forecast[-1])
        #         except Exception as e:
        #             print(f"ARIMA prediction failed: {e}")
        #             preds.append(np.nan)


# -------------------------------- Business Solution ---------------------------------------------------------------------------------

def select_best_model(results, weight_bias=0.4, weight_accuracy=0.4, weight_consistency=0.2):
    """
    Select the best model for each country based on weighted criteria, including consistency.

    Args:
        results (dict): The results dictionary containing backtesting metrics.
        weight_bias (float): Weight for bias in the final score.
        weight_accuracy (float): Weight for accuracy in the final score.
        weight_consistency (float): Weight for consistency (low standard deviation) in the final score.

    Returns:
        dict: Updated results dictionary with the best models for each country.
    """
    backtesting_results = results["backtesting_results"]
    #print("BACKTESTING", backtesting_results)
    best_models = {}

    for country, country_data in backtesting_results.items():
        #print(f"\nSelecting the best model for {country}...")
        model_scores = {}
        #print("WHAT IS IN THRE", country_data['predictions'])
        # Access the 'metrics' dictionary for the country
        all_model_metrics = country_data["metrics"]

        for model_name, model_metrics in all_model_metrics.items():
            
            try:
                # Extract the mean and standard deviation over cycles
                #avg_bias = np.nanmean(metrics.get("bias", []))
                #print(model_metrics)
                bias_values = [
                    d['bias'] 
                    for cycle_data in model_metrics.values()  # Iterate over cycle-level data
                    for d in cycle_data['metrics']           # Iterate over the list of dictionaries in 'metrics'
                    if 'bias' in d                           # Ensure 'bias' exists in the dictionary
                    ]

                avg_bias = np.nanmean(bias_values)
                std_bias = np.nanstd(bias_values)

                accuracy_values = [
                    d['accuracy'] 
                    for cycle_data in model_metrics.values()  # Iterate over cycle-level data
                    for d in cycle_data['metrics']           # Iterate over the list of dictionaries in 'metrics'
                    if 'accuracy' in d                           # Ensure 'accuracy' exists in the dictionary
                    ]

                avg_accuracy = np.nanmean(accuracy_values)
                std_accuracy = np.nanstd(accuracy_values)

                # Calculate consistency as the sum of standard deviations
                consistency = std_bias + std_accuracy

                # Calculate the weighted score
                # Lower bias and MAPE are better, higher accuracy is better
                score = (
                    (1 / abs(avg_bias)) * weight_bias + #1/|bias| to penalize large biases
                    avg_accuracy * weight_accuracy - 
                    consistency * weight_consistency  #Lower consistency penalty
                )

                model_scores[model_name] = score
                #print(f"Model: {model_name}, Score: {score:.2f}")
            except Exception as e:
                print(f"Error calculating score for model {model_name} in {country}: {e}")
                model_scores[model_name] = float("-inf")  # Penalize invalid models

        

        # Select the model with the highest score
        best_model = max(model_scores, key=model_scores.get)
        
        
        predictions = country_data['predictions'][best_model]
        # Extract the lists from the dictionary
        lists = list(predictions.values())

        # Compute the element-wise average
        averaged_predictions = [sum(values) / len(values) for values in zip(*lists)]
        #print("list", averaged_predictions)
        
        # Store the best model and its predictions
        best_models[country] = {
            "model": best_model,
            "score": model_scores[best_model],
            "predictions": averaged_predictions
        }
        #print(f"Best model for {country}: {best_model} with prediction: {predictions}  with score {model_scores[best_model]:.2f}")

    # Save the best models to the results dictionary
    results["best_models"] = best_models
    return results


# def select_best_model(results, weight_bias=0.4, weight_accuracy=0.4, weight_mape=0.2):
#     """
#     Select the best model for each country based on weighted criteria from backtesting results.

#     Args:
#         results (dict): The results dictionary containing backtesting metrics.
#         weight_bias (float): Weight for bias in the final score.
#         weight_accuracy (float): Weight for accuracy in the final score.
#         weight_mape (float): Weight for MAPE in the final score.

#     Returns:
#         dict: Updated results dictionary with the best models for each country.
#     """
#     backtesting_results = results.get("backtesting_results", {})
#     best_models = {}

#     for country, models in backtesting_results.items():
#         print(f"\nSelecting the best model for {country}...")
#         model_scores = {}

#         for model_name, metrics in models.items():
#             try:
#                 # Extract the mean values over cycles
#                 avg_bias = np.nanmean(metrics.get("bias", []))
#                 avg_accuracy = np.nanmean(metrics.get("accuracy", []))
#                 avg_mape = np.nanmean(metrics.get("mape", []))

#                 # Calculate the weighted score
#                 # Lower bias and MAPE are better, higher accuracy is better
#                 score = (
#                     (1 / abs(avg_bias)) * weight_bias + #1/|bias| to penalize large biases
#                     avg_accuracy * weight_accuracy - 
#                     avg_mape * weight_mape
#                 )

#                 model_scores[model_name] = score
#                 print(f"Model: {model_name}, Score: {score:.2f}")
#             except Exception as e:
#                 print(f"Error calculating score for model {model_name} in {country}: {e}")
#                 model_scores[model_name] = float("-inf")  # Penalize invalid models

#         # Select the model with the highest score
#         best_model = max(model_scores, key=model_scores.get)
#         best_models[country] = {
#             "model": best_model,
#             "score": model_scores[best_model]
#         }
#         print(f"Best model for {country}: {best_model} with score {model_scores[best_model]:.2f}")

#     # Save the best models to the results dictionary
#     results["best_models"] = best_models
#     return results







# -------------------------------- Pipeline Model ---------------------------------------------------------------------------------




def full_model_evaluation_pipeline(combined_df, shock_years, shock_quarter, shock_features, shock_magnitude, cycles=3):
    results = {}

    target_column = 'NET Claims Incurred'
    train_start, train_end = "2016-07-01", "2021-12-31"
    val_start, val_end = "2022-01-01", "2023-03-31"
    blind_test_start, blind_test_end = "2023-04-01", "2024-03-31"
    n_lags = 4 

    
    (X_train, y_train), (X_val, y_val), (X_blind_test, y_blind_test) = split_data(combined_df, train_start, train_end, val_start, val_end, blind_test_start, blind_test_end, target_column)

    # Debugging statements
    #print(f"y_train size: {y_train.shape[0]}")  # y_train size: 22
    #print(f"y_val size: {y_val.shape[0]}")  #y_val size: 5
    #print(f"y_blind_test size: {y_blind_test.shape[0]}") #y_blind_test size: 4

    # Train and evaluate a single default ML models that is obtained on all countries
    xgb_model = XGBRegressor()
    lgb_model = LGBMRegressor(min_data_in_bin=1, min_data_in_leaf=1, random_state=42)
    #print(f'test before training: {xgb_model}')
    xgb_model, xgb_results = train_and_evaluate_model(xgb_model, X_train, y_train, X_val, y_val, X_blind_test, y_blind_test, combined_df, target_column)
    #print(f'test after training: {xgb_model}')
    lgb_model, lgb_results = train_and_evaluate_model(lgb_model, X_train, y_train, X_val, y_val, X_blind_test, y_blind_test, combined_df, target_column)

    # Metrics for both single default ML models that is obtained on all countries on validation and blind test set
    results['metrics'] = {
        'Default XGBoost': xgb_results['metrics'],
        'Default LightGBM': lgb_results['metrics']
        }
    
    #print(results['metrics']['Default XGBoost'])
    #print("-----------------", results['metrics']['Default LightGBM'])


    # Save all-country, validation, and blind test predictions for default models
    results['predictions'] = {
        'all_countries_predictions': {
            'Default XGBoost': xgb_results['all_countries_predictions'],
            'Default LightGBM': lgb_results['all_countries_predictions']
        },
        'validation_predictions': {
            'Default XGBoost': xgb_results['validation_predictions'],
            'Default LightGBM': lgb_results['validation_predictions']
        },
        
        'blind_test_predictions': {
            'Default XGBoost': xgb_results['blind_test_predictions'],
            'Default LightGBM': lgb_results['blind_test_predictions']
        }
        
    }

    results['actuals'] = {
        
        'validation_actuals': {
            'Default XGBoost': xgb_results['validation_actuals'].reset_index(drop=True),
            'Default LightGBM': lgb_results['validation_actuals'].reset_index(drop=True)
        },
        'blind_test_actuals': {
            'Default XGBoost': xgb_results['blind_test_actuals'].reset_index(drop=True),
            'Default LightGBM': lgb_results['blind_test_actuals'].reset_index(drop=True)
        }
    }

    # Retraining and re-evaluation a single default ML models that is obtained on all countries
    xgb_trial_params = {
        'n_estimators': ('suggest_int', 50, 300),
        'learning_rate': ('suggest_float', 0.01, 0.3),
        'max_depth': ('suggest_int', 3, 10),
        'subsample': ('suggest_float', 0.5, 1.0),
        'colsample_bytree': ('suggest_float', 0.5, 1.0),
        'reg_alpha': ('suggest_int', 0, 1),
        'reg_lambda': ('suggest_int', 0, 1)
    }


    lgb_trial_params = {
        'n_estimators': ('suggest_int', 50, 300),
        'learning_rate': ('suggest_float', 0.01, 0.3),
        'max_depth': ('suggest_int', 3, 10),
        'num_leaves': ('suggest_int', 20, 50),
        'subsample': ('suggest_float', 0.5, 1.0),
        'colsample_bytree': ('suggest_float', 0.5, 1.0),
        'reg_alpha': ('suggest_int', 0, 1),
        'reg_lambda': ('suggest_int', 0, 1),
        'min_data_in_bin': ('min_data_in_bin', 8, 10),
        'min_data_in_leaf': ('min_data_in_leaf', 8, 10)
    }

    #lgb_model = LGBMRegressor(min_data_in_bin=1, min_data_in_leaf=1, random_state=42)
    best_xgb_params = tune_model(XGBRegressor, X_train, y_train, X_val, y_val, xgb_trial_params)
    best_lgb_params = tune_model(LGBMRegressor, X_train, y_train, X_val, y_val, lgb_trial_params)

    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = pd.concat([y_train, y_val], ignore_index=True)

    # Debugging statements 
    #print(f"Final y_combined length before ARIMA: {len(y_combined)}")  #Final y_combined length before ARIMA: 27
    #print(f"Final y_combined indices before ARIMA:\n{y_combined.index}")   #Final y_combined indices before ARIMA:RangeIndex(start=0, stop=27, step=1)
    #print(f"Final y_combined head before ARIMA:\n{y_combined.head()}")  # correct values
    #print(f"y_combined size (expected 27): {y_combined.shape[0]}")   #y_combined size (expected 27): 27
    #print(f"Combined train and validation target size: {y_combined.shape[0]}")  #Combined train and validation target size: 27
    #print(f"Are there duplicates in combined_df? {combined_df.duplicated().any()}")   #Are there duplicates in combined_df? False

    

    re_xgb_model, re_xgb_res = retrain_and_evaluate(
        XGBRegressor, best_xgb_params, X_combined, y_combined, X_blind_test, y_blind_test, combined_df, target_column
    )

    re_xgb_metrics = re_xgb_res['metrics']
    re_xgb_test_preds = re_xgb_res['test_preds']
    re_xgb_actuals = re_xgb_res['test_actuals']
    re_xgb_all_preds = re_xgb_res['all_countries_predictions']

    re_lgb_model, re_lgb_res = retrain_and_evaluate(
        LGBMRegressor, best_lgb_params, X_combined, y_combined, X_blind_test, y_blind_test, combined_df, target_column
    )

    re_lgb_metrics = re_lgb_res['metrics']
    re_lgb_test_preds = re_lgb_res['test_preds']
    re_lgb_actuals = re_lgb_res['test_actuals']
    re_lgb_all_preds = re_lgb_res['all_countries_predictions']


    # Add retrained all-country predictions and actuals
    results['predictions']['blind_test_predictions']['Retrained XGBoost'] = re_xgb_test_preds
    results['predictions']['blind_test_predictions']['Retrained LightGBM'] = re_lgb_test_preds
    results['predictions']['all_countries_predictions']['Retrained XGBoost'] = re_xgb_all_preds
    results['predictions']['all_countries_predictions']['Retrained LightGBM'] = re_lgb_all_preds
    results['actuals']['blind_test_actuals']['Retrained XGBoost'] = re_xgb_actuals.reset_index(drop=True)
    results['actuals']['blind_test_actuals']['Retrained LightGBM'] = re_lgb_actuals.reset_index(drop=True)

    models = {
        'Default XGBoost': xgb_model,
        'Default LightGBM': lgb_model
    }
    retrained_models = {
        'Retrained XGBoost': re_xgb_model,
        'Retrained LightGBM': re_lgb_model
    }

    # Add ML models to results
    results['models'] = {
        'Default XGBoost': xgb_model,
        'Default LightGBM': lgb_model,
        'Retrained XGBoost': re_xgb_model,
        'Retrained LightGBM': re_lgb_model
    }
    results['metrics']['Retrained XGBoost'] = re_xgb_metrics
    results['metrics']['Retrained LightGBM'] = re_lgb_metrics
    


    #print("BBBBSSSSS")

    # Country-wise Model evaluations
    results['country_metrics'] = evaluate_models_by_country(
    models, retrained_models, combined_df, X_combined, y_combined, 
    target_column, val_start, val_end, blind_test_start, blind_test_end
)
   
    arima_results = evaluate_arima_model_by_country(combined_df, target_column)

    for country, country_results in arima_results.items():
        results['country_metrics'][country]["ARIMA"] = country_results

    ma_results = evaluate_ma_model_by_country(combined_df, target_column) #, blind_test_start, blind_test_end, window=2)

    for country, country_results in ma_results.items():
        results['country_metrics'][country]["Moving Average"] = country_results

    #print("WHAT THE FUCK")
    #arima_data = split_data_for_arima_ma(combined_df, train_start, train_end, blind_test_start, blind_test_end, target_column)
    #print(f"END OF EVALUATION ON COUNTRY")

    arima_models = {country: arima_results[country]["fitted_model"] for country in arima_results.keys()}
   # Define model_dict and model_types for backtesting
    model_dicts = [models, retrained_models] #, {'ARIMA': arima_models, 'Moving Average': {}}]
    model_types = {
        'Default XGBoost': 'xgb',
        'Retrained XGBoost': 'xgb',
        'Default LightGBM': 'lgb',
        'Retrained LightGBM': 'lgb',
        'ARIMA': 'arima',
        'Moving Average': 'ma'
    }
    # Add a print statement to summarize the models and their types
    #print("\nModel preparation completed.")
    # print("Model Dictionary contains the following:")
    # for i, model_group in enumerate(model_dict):
    #     if isinstance(model_group, dict):
    #         print(f"  Model Group {i+1} (Dictionary): {list(model_group.keys())}")
    #     else:
    #         print(f"  Model Group {i+1} (List): {len(model_group)} models")

    # print("\nModel Types Mapping:")
    # for model_name, model_type in model_types.items():
    #     print(f"  {model_name}: {model_type}")

    #print(f"CALLING BACKTESTING WITH CYCLES={cycles}, LAGS={n_lags}.")
    results['backtesting_results'] = run_backtest(model_dicts, model_types, combined_df=combined_df, target_column=target_column, cycles=cycles, n_lags=n_lags)
    
    # arima_data = split_data_for_arima_ma(combined_df
    #                                      , train_start="2017-10-01"
    #                                      , train_end="2023-03-31"
    #                                      , blind_test_start="2023-04-01"
    #                                      , blind_test_end="2024-03-31"
    #                                      , target_column=target_column)
    
    # country_data = arima_data["Sweden"]
    # #print("country", country_data)
    # country_dfs = prepare_for_arima_ma(combined_df, target_column)
    # #print("dfs", country_dfs)
    # y_train_arima = country_dfs["Sweden"][target_column]
    # #print("y_train_arima", y_train_arima)
    # #model = arima_models["Sweden"]
    # #print("model", model)
    # model_type = "arima"
    #arima_all_cycle_metrics, arima_all_preds = arima_backtest_model_by_country(country_data, y_train_arima, model, model_type, target_column, {}, {}, window=22, cycles=3, n_lags=4)
    #arima_all_cycle_metrics, arima_all_preds = arima_backtest_model(arima_data, arima_models, model_type, target_column, {}, {}, window=22, cycles=3, n_lags=4)
    
    #print("testing arima back testing", arima_all_cycle_metrics, arima_all_preds)

    # Add a print statement to confirm and summarize backtesting results
    #print("\nBacktesting completed successfully.")
    # if 'backtesting_results' in results and results['backtesting_results']:
    #     print(f"Backtesting results stored. Number of models backtested: {len(results['backtesting_results'])}")
    # else:
    #     print("No backtesting results were produced. Please check the inputs and model configurations.")

    results['time_series_models'] = {
        'ARIMA': arima_models,
        'Moving Average': {}
    }

    # results['stressed_metrics'] = apply_stress_to_dataframe(
    #     combined_df, shock_years, shock_quarter, shock_features, shock_magnitude, 
    #     models, retrained_models, target_column, val_start, val_end, blind_test_start, blind_test_end
    # )
    #print("WHAT THE FUCK")
    # Select the best model for each country
    results = select_best_model(results, weight_bias=0.4, weight_accuracy=0.4, weight_consistency=0.2)
    #print("Best Model Selection:", results.get("best_models"))
    #print(results)  # Check if `results` is None or has the expected structure
    #print(results.keys())  # Check if 'country_metrics' is part of the keys

     # Specify the number of rows to select
    #print("prediction_periods")
    #print("THIS", combined_df['Year'])
    #print("THIS", combined_df['Year'][-n_lags:])
    #print("GO", combined_df['Quarter'])
    #print("GO", combined_df['Quarter'][-n_lags:])

    # Create a list of the last X elements from two columns concatenated with a space (last tab)
    results["prediction_periods"] = (combined_df['Year'][-n_lags:].astype(str) + ' Q' + combined_df['Quarter'][-n_lags:].astype(str)).tolist()
    
    
    #print("prediction_periods", results["prediction_periods"])
    # Pretty-print using json.dumps
    #print(json.dumps(results, indent=4))

    # Use pprint to print the dictionary
    #pprint.pprint(results)
    #print('predictions for def xgb',results['predictions']['blind_test_predictions']['Default XGBoost'])
    #print('actuals for def xgb',results['actuals']['blind_test_actuals']['Default XGBoost'])
    return results


# -------------------------------- Saving results ---------------------------------------------------------------------------------



def get_or_generate_results(combined_df, cycles=3):
    results_file = 'results.pkl'

    shock_years = [2017, 2019, 2020, 2023]
    shock_quarter = 4
    shock_features = [
        'NET Premiums Written', 'NET Premiums Earned', 'NET Claims Incurred',
        'Changes in other technical provisions', 'Expenses incurred', 
        'Total technical expenses', 'Other Expenses'
    ]
    shock_magnitude = {
        'NET Premiums Written': 0.1,
        'NET Premiums Earned': 0.1,
        'NET Claims Incurred': 0.20,
        'Changes in other technical provisions': 0.15,
        'Expenses incurred': 0.1,
        'Total technical expenses': 0.15,
        'Other Expenses': 0.2
    }

    
    # Check if the results file exists
    if os.path.exists(results_file):
        results = joblib.load(results_file)
    else:
        # Generate results and save if file doesn't exist
        results = full_model_evaluation_pipeline(combined_df, shock_years, shock_quarter, shock_features, shock_magnitude, cycles=cycles)
        joblib.dump(results, results_file)
    
    return results












#original

# # ---------------------------------- Back Testing ----------------------------------------------------------------

# # # model_dicts is models
# # # model_types is the dictionary defining each model's type

# def run_backtest(model_dicts, model_types, combined_df, target_column, cycles):
#     """
#     Run backtesting for all models (ML, ARIMA, MA) on the combined dataset.

#     Args:
#         model_dicts (list): List of model dictionaries (e.g., ML models, retrained models).
#         model_types (dict): Dictionary mapping model names to their types (e.g., 'xgb', 'arima', 'ma').
#         combined_df (pd.DataFrame): Combined dataset with all countries.
#         target_column (str): The target column for predictions.
#         cycles (int): Number of backtesting cycles.

#     Returns:
#         dict: Backtesting results for each country and model.
#     """
#     # Combine all ML models and types into one dictionary
#     all_models = {}
#     for model_dict in model_dicts:
#         all_models.update(model_dict)

#     countries = combined_df['Country'].unique()
#     backtesting_results = {}

#     for country in countries:
#         print(f"\nProcessing country in back testing: {country}")  # Debugging statement
#         country_data = combined_df[combined_df['Country'] == country]

#         # Preprocess data for ARIMA/MA
#         arima_df = prepare_for_arima_ma(country_data, target_column)

#         # Drop the first 5 rows of each country's data for ARIMA
#         arima_df = arima_df.iloc[5:]
#         print(f"arima_df after dropping the first 5 rows for ARIMA:\n{arima_df.head()}")

#         y_train_arima = arima_df[target_column]

#         model_results = {}
#         for model_name, model in all_models.items():
#             model_type = model_types[model_name]
#             print(f"Back testing evaluating model: {model_name} (Type: {model_type})")  # Debugging statement
#             backtest_metrics = backtest_model(country_data, y_train_arima, model, model_type, target_column, cycles=cycles, future_forecast_steps=4)
#             model_results[model_name] = backtest_metrics

#             # Save backtesting results for this model
#             #model_results[model_name] = backtest_metrics

#         print(f"Completed backtesting for {country}.")  # Debugging statement
#         backtesting_results[country] = model_results

#     print("Backtesting complete.")  # Debugging statement
#     return backtesting_results


# def backtest_model(country_data, y_train_arima, model, model_type, target_column, window=26, test_size=5, cycles=3, future_forecast_steps=4):
#     """
#     Perform backtesting for a single model (ML, ARIMA, MA) on a country's data.

#     Args:
#         country_data (pd.DataFrame): Full dataset for the specific country.
#         arima_df (pd.DataFrame): Preprocessed dataset for ARIMA/MA models.
#         model (object): The model to be backtested (e.g., ML, ARIMA, MA).
#         model_type (str): The type of the model ('xgb', 'arima', 'ma').
#         target_column (str): The target column for predictions.
#         window (int): Training window size.
#         test_size (int): Testing window size.
#         cycles (int): Number of backtesting cycles.

#     Returns:
#         dict: Metrics for each backtesting cycle.
#     """
#     print(f"Starting backtest for model type: {model_type}")  # Debugging statement
#     cycle_metrics = {'bias': [], 'accuracy': [], 'mape': [], 'coc': []}
#     prev_cycle_preds = None

#     # Determine model type
#     is_ml_model = model_type in ['xgb', 'lgb']
#     is_arima_model = model_type == 'arima'
#     is_ma_model = model_type == 'ma'

#     # Define blind test data. This line ensures that the blind test data (y_blind_test_country) is extracted consistently across all model types (ML, ARIMA, MA) without being affected by the ARIMA-specific preprocessing
#     y_blind_test_country = country_data[target_column].iloc[-test_size:]

#     # ARIMA and MA models if applicable
#     if is_arima_model:
#         print("Back testing training ARIMA model...")  # Debugging statement
#         arima_model, arima_blind_test_forecast, _ = train_arima_model(
#             y_combined=y_train_arima,  # Combined training data for ARIMA
#             blind_test_data=y_blind_test_country,  # Blind test set for the specific country
#             blind_test_steps=len(y_blind_test_country),  # Number of steps for the blind test forecast
#             future_forecast_steps=future_forecast_steps  # Number of future steps to forecast
#         )
#     elif is_ma_model:
#         print("Back testing training MA model...")  # Debugging statement
#         ma_model, ma_blind_test_forecast, _ = train_ma_model(
#             y_combined=y_train_arima,
#             blind_test_data=country_data[target_column].iloc[window:],
#             window=4,
#             forecast_steps=test_size
#         )
#     if is_arima_model or is_ma_model:
#         window = 20
#     for i in range(cycles):
#         print(f"Cycle {i + 1}/{cycles}...")  # Debugging statement
#         b_train = country_data.iloc[0:window - i]
#         b_test = country_data.iloc[window - i:]
#         test_size = len(b_test)

#         b_X_train = b_train.drop([target_column, 'Date', 'Country'], axis=1, errors='ignore')
#         b_y_train = b_train[target_column]
#         b_X_test = b_test.drop([target_column, 'Date', 'Country'], axis=1, errors='ignore')
#         b_y_test = b_test[target_column]

#         if is_ml_model:
#             print(f"Back testing training ML model...")  # Debugging statement
#             model.fit(b_X_train, b_y_train)
#             preds = model.predict(b_X_test)
            
#         elif is_arima_model:
#             try:
#                 print("Back testing: Forecasting with ARIMA model...")  # Debugging statement
#                 preds = arima_blind_test_forecast[-test_size:]  # Use ARIMA model for forecasting
#             except Exception as e:
#                 print(f"ARIMA model forecasting failed: {e}")
#                 preds = np.full(len(b_y_test), np.nan)


#         elif is_ma_model:
#             try:
#                 print("Back testing: Forecasting with MA model...")  # Debugging statement
#                 preds = ma_blind_test_forecast[-test_size:]   # Use MA model for forecasting
#             except Exception as e:
#                 print(f"MA model forecasting failed: {e}")
#                 preds = np.full(len(b_y_test), np.nan)

        
#         # Ensure predictions align with test data
#         if len(preds) != len(b_y_test):
#             print(f"Prediction length mismatch: preds={len(preds)}, b_y_test={len(b_y_test)}")  # Debugging statement
#             if len(preds) == 0:
#                 # Handle empty predictions by filling with mean of training set
#                 print("Warning: Predictions are empty. Filling with mean of training data.")
#                 preds = np.full(len(b_y_test), b_y_train.mean())

#             elif len(preds) > len(b_y_test):
#                 # Truncate predictions to match test set length
#                 print("Warning: Predictions are longer than test set. Truncating to match length.")
#                 preds = preds[:len(b_y_test)]

#             elif len(preds) < len(b_y_test):
#                 # Extend predictions with the last value to match test set length
#                 print("Warning: Predictions are shorter than test set. Padding with last prediction value.")
#                 padding = [preds[-1]] * (len(b_y_test) - len(preds))
#                 preds = np.append(preds, padding)


#         # Calculate metrics if predictions are valid
#         if len(preds) == len(b_y_test) and not np.isnan(preds).all():
#             bias = (np.mean(preds - b_y_test) / np.mean(b_y_test)) * 100
#             accuracy = 100 - mean_absolute_percentage_error(b_y_test, preds) * 100
#             mape = mean_absolute_percentage_error(b_y_test, preds) * 100

#             coc = ((np.mean(preds - prev_cycle_preds) / np.mean(prev_cycle_preds)) * 100
#                    if prev_cycle_preds is not None and len(preds) == len(prev_cycle_preds) else np.nan)

#             cycle_metrics['bias'].append(bias)
#             cycle_metrics['accuracy'].append(accuracy)
#             cycle_metrics['mape'].append(mape)
#             cycle_metrics['coc'].append(coc)
#             prev_cycle_preds = preds

#             print(f"Cycle {i + 1} metrics: Bias={bias:.2f}, Accuracy={accuracy:.2f}, MAPE={mape:.2f}")  # Debugging statement

#     print("Backtesting complete.")  # Debugging statement
#     return cycle_metrics



# ------------------------------------- Apply Stress Testing -------------------------------------------------------

# def apply_stress_to_dataframe(combined_df, shock_years, shock_quarter, shock_features, shock_magnitude, models, retrained_models, target_column, val_start, val_end, blind_test_start, blind_test_end):
#     combined_df_stressed = combined_df.copy()

#     shock_years = [2017, 2019, 2020, 2023]
#     shock_quarter = 4
#     shock_features = [
#         'NET Premiums Written', 'NET Premiums Earned', 'NET Claims Incurred',
#         'Changes in other technical provisions', 'Expenses incurred', 
#         'Total technical expenses', 'Other Expenses'
#     ] + [col for col in combined_df.columns if '_Lag' in col]

#     shock_magnitude = {
#         'NET Premiums Written': 0.1,
#         'NET Premiums Earned': 0.1,
#         'NET Claims Incurred': 0.20,
#         'Changes in other technical provisions': 0.15,
#         'Expenses incurred': 0.1,
#         'Total technical expenses': 0.15,
#         'Other Expenses': 0.2
#     }

    
#     # Apply shocks to the stressed dataset
#     for year in shock_years:
#         for feature in shock_features:
#             if feature in shock_magnitude:  # Apply specified shock for known features
#                 magnitude = 1 + shock_magnitude[feature]
#             else:  # Default shock for lagged or unknown features
#                 magnitude = 1.1

#             # Apply the shock to the selected period and feature
#             combined_df_stressed.loc[
#                 (combined_df_stressed['Year'] == year) & 
#                 (combined_df_stressed['Quarter'] == shock_quarter), feature
#             ] *= magnitude

#     # Initialize storage for stressed metrics
#     stressed_metrics = {}

#     for country in combined_df['Country'].unique():
#         print(f"\nEvaluating stressed metrics for country: {country}")
#         stressed_val_data = combined_df_stressed[
#             (combined_df_stressed['Country'] == country) &
#             (combined_df_stressed['Date'] >= val_start) &
#             (combined_df_stressed['Date'] <= val_end)
#         ]
#         stressed_blind_test_data = combined_df_stressed[
#             (combined_df_stressed['Country'] == country) &
#             (combined_df_stressed['Date'] >= blind_test_start) &
#             (combined_df_stressed['Date'] <= blind_test_end)
#         ]
#         stressed_X_val = stressed_val_data.drop(columns=[target_column, 'Date', 'Country'])
#         stressed_y_val = stressed_val_data[target_column]
#         stressed_X_blind_test = stressed_blind_test_data.drop(columns=[target_column, 'Date', 'Country'])
#         stressed_y_blind_test = stressed_blind_test_data[target_column]

#         # Preprocess ARIMA-specific dataset
#         arima_df = prepare_for_arima_ma(combined_df_stressed[combined_df_stressed['Country'] == country], target_column)
#         y_train_arima = arima_df[target_column][:len(combined_df_stressed)]  # Slice to match combined data length

#         country_results = {}

#         # Default ML models on stressed validation and blind test sets
#         for model_name, model in models.items():
#             model.fit(stressed_X_val, stressed_y_val)
#             stressed_val_preds = model.predict(stressed_X_val)
#             stressed_blind_test_preds = model.predict(stressed_X_blind_test)


#         country_results[model_name] = {
#                 'validation': {
#                     'MAPE%': mean_absolute_percentage_error(stressed_y_val, stressed_val_preds) * 100,
#                     'Accuracy%': 100 - mean_absolute_percentage_error(stressed_y_val, stressed_val_preds) * 100,
#                     'Bias%': (np.mean(stressed_val_preds - stressed_y_val) / np.mean(stressed_y_val)) * 100
#                 },
#                 'blind_test': {
#                     'MAPE%': mean_absolute_percentage_error(stressed_y_blind_test, stressed_blind_test_preds) * 100,
#                     'Accuracy%': 100 - mean_absolute_percentage_error(stressed_y_blind_test, stressed_blind_test_preds) * 100,
#                     'Bias%': (np.mean(stressed_blind_test_preds - stressed_y_blind_test) / np.mean(stressed_y_blind_test)) * 100
#                 }
#             }

#         # Retrained ML models on stressed blind test set only
#         for model_name, model in retrained_models.items():
#             model.fit(stressed_X_blind_test, stressed_y_blind_test)
#             stressed_re_blind_test_preds = model.predict(stressed_X_blind_test)

#             country_results[model_name] = {
#                 'blind_test': {
#                     'MAPE%': mean_absolute_percentage_error(stressed_y_blind_test, stressed_re_blind_test_preds) * 100,
#                     'Accuracy%': 100 - mean_absolute_percentage_error(stressed_y_blind_test, stressed_re_blind_test_preds) * 100,
#                     'Bias%': (np.mean(stressed_re_blind_test_preds - stressed_y_blind_test) / np.mean(stressed_y_blind_test)) * 100
#                 }
#             }

        
#         # Use ARIMA and MA models
#         try:
#             _, stressed_arima_forecast, _ = train_arima_model(
#                 y_combined=y_train_arima,
#                 blind_test_data=stressed_y_blind_test,
#                 forecast_steps=len(stressed_y_blind_test)
#             )

#             # Store metrics for ARIMA and MA models
#             country_results['ARIMA'] = {
#                 'blind_test': {
#                     'MAPE%': mean_absolute_percentage_error(stressed_y_blind_test, stressed_arima_forecast) * 100,
#                     'Accuracy%': 100 - mean_absolute_percentage_error(stressed_y_blind_test, stressed_arima_forecast) * 100,
#                     'Bias%': (np.mean(stressed_arima_forecast - stressed_y_blind_test) / np.mean(stressed_y_blind_test)) * 100
#                 }
#             }
#         except Exception as e:
#             print(f"ARIMA model failed for {country}: {e}")

#         try:
#             _, stressed_ma_forecast, _ = train_ma_model(
#                 y_combined=y_train_arima,
#                 blind_test_data=stressed_y_blind_test,
#                 window=4,
#                 forecast_steps=len(stressed_y_blind_test)
#             )

#             country_results['Moving Average'] = {
#                 'blind_test': {
#                     'MAPE%': mean_absolute_percentage_error(stressed_y_blind_test, stressed_ma_forecast) * 100,
#                     'Accuracy%': 100 - mean_absolute_percentage_error(stressed_y_blind_test, stressed_ma_forecast) * 100,
#                     'Bias%': (np.mean(stressed_ma_forecast - stressed_y_blind_test) / np.mean(stressed_y_blind_test)) * 100
#                 }
#             }
#         except Exception as e:
#             print(f"Moving Average model failed for {country}: {e}")

#         # Add the results for this country to the stressed_metrics dictionary
#         stressed_metrics[country] = country_results

#     return stressed_metrics
