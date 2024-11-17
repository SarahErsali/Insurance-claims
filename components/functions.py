import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import shap
from math import sqrt
import optuna
from statsmodels.tsa.arima.model import ARIMA
from statsforecast.models import AutoARIMA
from sktime.forecasting.arima import AutoARIMA
from pmdarima import auto_arima
import matplotlib
matplotlib.use('Agg')  # Switch to a non-interactive backend
import matplotlib.pyplot as plt
import os
import joblib
from components.data import prepare_for_arima_ma





# ------------------------------------- Data splitting -------------------------------------------------------


def split_data(combined_df, train_start, train_end, val_start, val_end, blind_test_start, blind_test_end, target_column):

    train_data = combined_df[(combined_df['Date'] >= train_start) & (combined_df['Date'] <= train_end)]
    val_data = combined_df[(combined_df['Date'] >= val_start) & (combined_df['Date'] <= val_end)]
    blind_test_data = combined_df[(combined_df['Date'] >= blind_test_start) & (combined_df['Date'] <= blind_test_end)]


    X_train = train_data.drop(columns=[target_column, 'Date', 'Country'])
    y_train = train_data[target_column]
    X_val = val_data.drop(columns=[target_column, 'Date', 'Country'])
    y_val = val_data[target_column]
    X_blind_test = blind_test_data.drop(columns=[target_column, 'Date', 'Country'])
    y_blind_test = blind_test_data[target_column]

    return (X_train, y_train), (X_val, y_val), (X_blind_test, y_blind_test)  # Return tuples


# ---------------------------------- Default Model Developement -----------------------------------------------------



def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_blind_test=None, y_blind_test=None, combined_df=None, target_column=None):
    """
    Train and evaluate the model. Optionally, make predictions on the entire combined dataset.
    
    Args:
        combined_df (pd.DataFrame): The full combined dataset (optional, for all-country predictions).
        target_column (str): The name of the target column (optional).
    """
    
    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    metrics = {
        'validation': {
            'MAPE%': mean_absolute_percentage_error(y_val, val_preds) * 100,
            'Accuracy%': 100 - mean_absolute_percentage_error(y_val, val_preds) * 100,
            'Bias%': (np.mean(val_preds - y_val) / np.mean(y_val)) * 100
        }
    }
    blind_test_preds = None
    if X_blind_test is not None and y_blind_test is not None:
        blind_test_preds = model.predict(X_blind_test)
        metrics['blind_test'] = {
            'MAPE%': mean_absolute_percentage_error(y_blind_test, blind_test_preds) * 100,
            'Accuracy%': 100 - mean_absolute_percentage_error(y_blind_test, blind_test_preds) * 100,
            'Bias%': (np.mean(blind_test_preds - y_blind_test) / np.mean(y_blind_test)) * 100
        }

    # Add predictions for all countries if combined_df and target_column are provided
    all_countries_preds = None
    if combined_df is not None and target_column is not None:
        X_all = combined_df.drop(columns=[target_column, 'Date', 'Country'], errors='ignore')
        all_countries_preds = model.predict(X_all)

    return metrics, val_preds, blind_test_preds, all_countries_preds


# ---------------------------------- Model Optimization (Bayesian Method) -------------------------------------------


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

            # For LightGBM-specific parameter if using LightGBM trials
            if 'num_leaves' in trial_params:
                params['num_leaves'] = trial.suggest_int('num_leaves', 20, 50)
        
            #print("Trial parameters generated:", params)

            # Dynamically add verbose=-1 if the model is LightGBM to suppress repetative warnings in the console
            if model_class.__name__ == 'LGBMClassifier' or model_class.__name__ == 'LGBMRegressor':
                params['verbose'] = -1
        
            # Create and train model with these parameters
            model = model_class(**params)
            model.fit(X_train, y_train)
            val_preds = model.predict(X_val)
            return sqrt(mean_squared_error(y_val, val_preds))
        
        except TypeError as e:
            print(f"Error in parameter setting: {e}")
            raise e  # Re-raise to capture Optuna logging


    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    return study.best_params



# ----------------------------------- Retraining on Blind Test Set -----------------------------------------


def retrain_and_evaluate(model_class, best_params, X_combined, y_combined, X_blind_test, y_blind_test, combined_df=None, target_column=None):
    """
    Retrain the model on combined data and evaluate on blind test set. Optionally predict for all countries.
    
    Args:
        combined_df (pd.DataFrame): Full combined dataset (optional, for all-country predictions).
        target_column (str): Target column name (optional).
    """
    # Dynamically add verbose=-1 if the model is LightGBM to suppress repetative warnings in the console
    if model_class.__name__ == 'LGBMClassifier' or model_class.__name__ == 'LGBMRegressor':
        best_params['verbose'] = -1

    model = model_class(**best_params)
    model.fit(X_combined, y_combined)
    test_preds = model.predict(X_blind_test)
    metrics = {
        'blind_test': {
            'MAPE%': mean_absolute_percentage_error(y_blind_test, test_preds) * 100,
            'Accuracy%': 100 - mean_absolute_percentage_error(y_blind_test, test_preds) * 100,
            'Bias%': (np.mean(test_preds - y_blind_test) / np.mean(y_blind_test)) * 100
        }
    }

    # Predict for all countries if combined_df and target_column are provided
    all_countries_preds = None
    if combined_df is not None and target_column is not None:
        X_all = combined_df.drop(columns=[target_column, 'Date', 'Country'], errors='ignore')
        all_countries_preds = model.predict(X_all)

    return model, metrics, test_preds, all_countries_preds



# ----------------------------------- Developing Time Series Model -----------------------------------------

def train_arima_model(y_combined, blind_test_data, forecast_steps, max_p=5, max_q=5, max_d=2):
    """
    Train an ARIMA model on the provided target series and forecast future values.

    Args:
        y_combined (pd.Series): Combined target time series (train + validation) for ARIMA training.
        blind_test_data (pd.Series): Target series for the blind test set.
        forecast_steps (int): Number of future steps to forecast.
        max_p (int): Maximum order for the AR term.
        max_q (int): Maximum order for the MA term.
        max_d (int): Maximum order for differencing.

    Returns:
        tuple: (ARIMA model, blind test forecast array, future forecast array)
    """
    # Ensure y_combined has no NaN values
    if y_combined.isnull().any():
        print("Warning: NaN values detected in y_combined. Please ensure it's cleaned before passing to ARIMA.")
        return None, np.full(len(blind_test_data), np.nan), np.full(forecast_steps, np.nan)

    if len(y_combined) < max_p + max_d + max_q + 1:
        print("Warning: Insufficient data points for ARIMA model fitting.")
        return None, np.full(len(blind_test_data), np.nan), np.full(forecast_steps, np.nan)

    try:
        # Automatically determine best ARIMA parameters
        arima_params_model = auto_arima(
            y_combined, seasonal=False, stepwise=True, trace=False,
            error_action='ignore', max_p=max_p, max_q=max_q, max_d=max_d
        )
        best_params = arima_params_model.order

        arima_model = ARIMA(y_combined, order=best_params)
        arima_model_fitted = arima_model.fit()

        arima_forecast = arima_model_fitted.forecast(steps=len(blind_test_data))
        future_arima_forecast = arima_model_fitted.forecast(steps=forecast_steps)

        return arima_model_fitted, arima_forecast, future_arima_forecast
    except Exception as e:
        print(f"Warning: ARIMA model fitting or forecasting failed: {e}")
        return None, np.full(len(blind_test_data), np.nan), np.full(forecast_steps, np.nan)  # Return NaN forecast if fitting fails
    



def train_ma_model(y_combined, blind_test_data, window, forecast_steps):
    """
    Train a simple Moving Average (MA) model on the provided target series and forecast future values.

    Args:
        y_combined (pd.Series): Combined target time series (train + validation) for MA training.
        blind_test_data (pd.Series): Target series for the blind test set.
        window (int): Rolling window size for the moving average.
        forecast_steps (int): Number of future steps to forecast.

    Returns:
        tuple: (Moving Average series, blind test forecast array, future forecast array)
    """
    # Ensure y_combined is valid
    if y_combined.isnull().any():
        print("Warning: NaN values detected in y_combined. Please ensure it's cleaned before passing to MA model.")
        return None, np.full(len(blind_test_data), np.nan), np.full(forecast_steps, np.nan)

    if len(y_combined) < window:
        print(f"Warning: Not enough data points to calculate moving average with window={window}.")
        return None, np.full(len(blind_test_data), np.nan), np.full(forecast_steps, np.nan)

    try:
        # Calculate the rolling mean
        ma_model = y_combined.rolling(window=window).mean()

        # Ensure there's enough data in the rolling mean. Forecast for blind test: Use the last known rolling mean
        if len(ma_model.dropna()) < 1:
            print("Warning: Not enough non-NaN values in the rolling mean.")
            return None, np.full(len(blind_test_data), np.nan), np.full(forecast_steps, np.nan)

        # Use the last rolling mean for blind test and future forecast
        last_known_mean = ma_model.dropna().iloc[-1]
        ma_forecast = np.full(len(blind_test_data), last_known_mean)  # Extend with last rolling mean
        future_ma_forecast = np.full(forecast_steps, last_known_mean)  # Extend future steps with the same mean

        return ma_model, ma_forecast, future_ma_forecast
    except Exception as e:
        print(f"Warning: MA model calculation or forecasting failed: {e}")
        return None, np.full(len(blind_test_data), np.nan), np.full(forecast_steps, np.nan)





# ----------------------------------- Model Evaluation on Each Country -----------------------------------------


def evaluate_models_by_country(models, retrained_models, combined_df, target_column, val_start, val_end, blind_test_start, blind_test_end, y_combined):
    countries = combined_df['Country'].unique()
    results = {}
    arima_models = {}
    ma_models = {}

    for country in countries:
        print(f"\nEvaluating models for country: {country}")
        country_data = combined_df[combined_df['Country'] == country]

        # Splitting data for the specific country
        (X_train_country, y_train_country), (X_val_country, y_val_country), (X_blind_test_country, y_blind_test_country) = split_data(
            country_data, train_start="2016-07-01", train_end="2021-12-31", 
            val_start=val_start, val_end=val_end, 
            blind_test_start=blind_test_start, blind_test_end=blind_test_end, 
            target_column=target_column
        )



        # Use prepare_for_arima_ma to clean the data for ARIMA and MA models
        arima_df = prepare_for_arima_ma(country_data, target_column)
        y_train_arima = arima_df[target_column][:len(y_train_country)]


        country_results = {}

        # Evaluate Default ML Models
        for model_name, model in models.items():
            try:
                val_preds = model.predict(X_val_country)
                blind_test_preds = model.predict(X_blind_test_country)

                metrics = {
                    'validation': {
                        'MAPE%': mean_absolute_percentage_error(y_val_country, val_preds) * 100,
                        'Accuracy%': 100 - mean_absolute_percentage_error(y_val_country, val_preds) * 100,
                        'Bias%': (np.mean(val_preds - y_val_country) / np.mean(y_val_country)) * 100
                    },
                    'blind_test': {
                        'MAPE%': mean_absolute_percentage_error(y_blind_test_country, blind_test_preds) * 100,
                        'Accuracy%': 100 - mean_absolute_percentage_error(y_blind_test_country, blind_test_preds) * 100,
                        'Bias%': (np.mean(blind_test_preds - y_blind_test_country) / np.mean(y_blind_test_country)) * 100
                    }
                }

                country_results[model_name] = {
                    'metrics': metrics,
                    'validation_predictions': val_preds,
                    'blind_test_predictions': blind_test_preds
                }
            except Exception as e:
                print(f"Error evaluating {model_name} for {country}: {e}")

        # Evaluate Retrained ML Models
        for model_name, model in retrained_models.items():
            try:
                blind_test_preds = model.predict(X_blind_test_country)

                metrics = {
                    'blind_test': {
                        'MAPE%': mean_absolute_percentage_error(y_blind_test_country, blind_test_preds) * 100,
                        'Accuracy%': 100 - mean_absolute_percentage_error(y_blind_test_country, blind_test_preds) * 100,
                        'Bias%': (np.mean(blind_test_preds - y_blind_test_country) / np.mean(y_blind_test_country)) * 100
                    }
                }

                country_results[model_name] = {
                    'metrics': metrics,
                    'blind_test_predictions': blind_test_preds
                }
            except Exception as e:
                print(f"Error evaluating retrained model {model_name} for {country}: {e}")

        # Evaluate ARIMA Model
        try:
            arima_model, arima_forecast, future_arima_forecast = train_arima_model(
                y_combined=arima_df[target_column],  # Use the full ARIMA-preprocessed target column
                blind_test_data=y_blind_test_country,  # Pass the blind test set
                forecast_steps=len(y_blind_test_country)  # Forecast steps equal to the length of the blind test set
            )

            arima_models[country] = arima_model
            arima_metrics = {
                'blind_test': {
                    'MAPE%': mean_absolute_percentage_error(y_blind_test_country, arima_forecast) * 100,
                    'Accuracy%': 100 - mean_absolute_percentage_error(y_blind_test_country, arima_forecast) * 100,
                    'Bias%': (np.mean(arima_forecast - y_blind_test_country) / np.mean(y_blind_test_country)) * 100
                }
            }
            country_results['ARIMA'] = {
                'metrics': arima_metrics,
                'blind_test_predictions': arima_forecast,
                'future_forecast': future_arima_forecast
            }
        except Exception as e:
            print(f"ARIMA model evaluation failed for {country}: {e}")

        # Evaluate Moving Average Model
        try:
            ma_model, ma_forecast, future_ma_forecast = train_ma_model(
                y_combined=arima_df[target_column],  # Use the full MA-preprocessed target column
                blind_test_data=y_blind_test_country,  # Pass the blind test set
                window=4,  # Rolling window size for MA
                forecast_steps=len(y_blind_test_country)  # Forecast steps equal to the length of the blind test set
            )

            ma_models[country] = ma_model
            ma_metrics = {
                'blind_test': {
                    'MAPE%': mean_absolute_percentage_error(y_blind_test_country, ma_forecast) * 100,
                    'Accuracy%': 100 - mean_absolute_percentage_error(y_blind_test_country, ma_forecast) * 100,
                    'Bias%': (np.mean(ma_forecast - y_blind_test_country) / np.mean(y_blind_test_country)) * 100
                }
            }
            country_results['Moving Average'] = {
                'metrics': ma_metrics,
                'blind_test_predictions': ma_forecast,
                'future_forecast': future_ma_forecast
            }
        except Exception as e:
            print(f"Moving Average model evaluation failed for {country}: {e}")

        results[country] = country_results

    return results, arima_models, ma_models







# ---------------------------------- Back Testing ----------------------------------------------------------------

# # model_dicts is models
# # model_types is the dictionary defining each model's type

def run_backtest(model_dicts, model_types, combined_df, target_column, cycles):
    """
    Run backtesting for all models (ML, ARIMA, MA) on the combined dataset.

    Args:
        model_dicts (list): List of model dictionaries (e.g., ML models, retrained models).
        model_types (dict): Dictionary mapping model names to their types (e.g., 'xgb', 'arima', 'ma').
        combined_df (pd.DataFrame): Combined dataset with all countries.
        target_column (str): The target column for predictions.
        cycles (int): Number of backtesting cycles.

    Returns:
        dict: Backtesting results for each country and model.
    """
    # Combine all models and types into one dictionary
    all_models = {}
    for model_dict in model_dicts:
        all_models.update(model_dict)

    countries = combined_df['Country'].unique()
    backtesting_results = {}

    for country in countries:
        country_data = combined_df[combined_df['Country'] == country]

        # Preprocess data for ARIMA/MA
        arima_df = prepare_for_arima_ma(country_data, target_column)

        model_results = {}
        for model_name, model in all_models.items():
            model_type = model_types[model_name]
            backtest_metrics = backtest_model(country_data, arima_df, model, model_type, target_column, cycles=cycles)
            model_results[model_name] = backtest_metrics

            # Save backtesting results for this model
            model_results[model_name] = backtest_metrics

        backtesting_results[country] = model_results

    return backtesting_results


def backtest_model(country_data, arima_df, model, model_type, target_column, window=24, test_size=12, cycles=3):
    """
    Perform backtesting for a single model (ML, ARIMA, MA) on a country's data.

    Args:
        country_data (pd.DataFrame): Full dataset for the specific country.
        arima_df (pd.DataFrame): Preprocessed dataset for ARIMA/MA models.
        model (object): The model to be backtested (e.g., ML, ARIMA, MA).
        model_type (str): The type of the model ('xgb', 'arima', 'ma').
        target_column (str): The target column for predictions.
        window (int): Training window size.
        test_size (int): Testing window size.
        cycles (int): Number of backtesting cycles.

    Returns:
        dict: Metrics for each backtesting cycle.
    """
    cycle_metrics = {'bias': [], 'accuracy': [], 'mape': [], 'coc': []}
    prev_cycle_preds = None

    # Determine model type
    is_ml_model = model_type in ['xgb', 'lgb']
    is_arima_model = model_type == 'arima'
    is_ma_model = model_type == 'ma'

    # Pre-train ARIMA and MA models if applicable
    if is_arima_model:
        arima_model, arima_blind_test_forecast, _ = train_arima_model(
            y_combined=arima_df[target_column], 
            blind_test_data=country_data[target_column].iloc[window:], 
            forecast_steps=test_size
        )
    elif is_ma_model:
        ma_model, ma_blind_test_forecast, _ = train_ma_model(
            y_combined=arima_df[target_column],
            blind_test_data=country_data[target_column].iloc[window:],
            window=4,
            forecast_steps=test_size
        )

    for i in range(cycles):
        b_train = country_data.iloc[i:i + window]
        b_test = country_data.iloc[i + window:i + window + test_size]

        b_X_train = b_train.drop([target_column, 'Date', 'Country'], axis=1, errors='ignore')
        b_y_train = b_train[target_column]
        b_X_test = b_test.drop([target_column, 'Date', 'Country'], axis=1, errors='ignore')
        b_y_test = b_test[target_column]

        if is_ml_model:
            model.fit(b_X_train, b_y_train)
            preds = model.predict(b_X_test)
            
        elif is_arima_model:
            try:
                preds = arima_blind_test_forecast[i:i + test_size]  # Use pre-trained ARIMA model for forecasting
            except Exception as e:
                print(f"ARIMA model forecasting failed: {e}")
                preds = np.full(len(b_y_test), np.nan)


        elif is_ma_model:
            try:
                preds = ma_blind_test_forecast[i:i + test_size]   # Use pre-trained MA model for forecasting
            except Exception as e:
                print(f"MA model forecasting failed: {e}")
                preds = np.full(len(b_y_test), np.nan)

        
        # Ensure predictions align with test data
        if len(preds) != len(b_y_test):
            if len(preds) == 0:
                # Handle empty predictions by filling with mean of training set
                print("Warning: Predictions are empty. Filling with mean of training data.")
                preds = np.full(len(b_y_test), b_y_train.mean())

            elif len(preds) > len(b_y_test):
                # Truncate predictions to match test set length
                print("Warning: Predictions are longer than test set. Truncating to match length.")
                preds = preds[:len(b_y_test)]

            elif len(preds) < len(b_y_test):
                # Extend predictions with the last value to match test set length
                print("Warning: Predictions are shorter than test set. Padding with last prediction value.")
                padding = [preds[-1]] * (len(b_y_test) - len(preds))
                preds = np.append(preds, padding)


        # Calculate metrics if predictions are valid
        if len(preds) == len(b_y_test) and not np.isnan(preds).all():
            bias = (np.mean(preds - b_y_test) / np.mean(b_y_test)) * 100
            accuracy = 100 - mean_absolute_percentage_error(b_y_test, preds) * 100
            mape = mean_absolute_percentage_error(b_y_test, preds) * 100

            coc = ((np.mean(preds - prev_cycle_preds) / np.mean(prev_cycle_preds)) * 100
                   if prev_cycle_preds is not None and len(preds) == len(prev_cycle_preds) else np.nan)

            cycle_metrics['bias'].append(bias)
            cycle_metrics['accuracy'].append(accuracy)
            cycle_metrics['mape'].append(mape)
            cycle_metrics['coc'].append(coc)
            prev_cycle_preds = preds

    return cycle_metrics



# ------------------------------------- Apply Stress Testing -------------------------------------------------------

def apply_stress_to_dataframe(combined_df, shock_years, shock_quarter, shock_features, shock_magnitude, models, retrained_models, target_column, val_start, val_end, blind_test_start, blind_test_end):
    combined_df_stressed = combined_df.copy()

    shock_years = [2017, 2020, 2023]
    shock_quarter = 4
    shock_features = [
        'NET Premiums Written', 'NET Premiums Earned', 'NET Claims Incurred',
        'Changes in other technical provisions', 'Expenses incurred', 
        'Total technical expenses', 'Other Expenses'
    ] + [col for col in combined_df.columns if '_Lag' in col]

    shock_magnitude = {
        'NET Premiums Written': 0.1,
        'NET Premiums Earned': 0.1,
        'NET Claims Incurred': 0.20,
        'Changes in other technical provisions': 0.15,
        'Expenses incurred': 0.1,
        'Total technical expenses': 0.15,
        'Other Expenses': 0.2
    }

    
    # Apply shocks to the stressed dataset
    for year in shock_years:
        for feature in shock_features:
            if feature in shock_magnitude:  # Apply specified shock for known features
                magnitude = 1 + shock_magnitude[feature]
            else:  # Default shock for lagged or unknown features
                magnitude = 1.1

            # Apply the shock to the selected period and feature
            combined_df_stressed.loc[
                (combined_df_stressed['Year'] == year) & 
                (combined_df_stressed['Quarter'] == shock_quarter), feature
            ] *= magnitude

    # Initialize storage for stressed metrics
    stressed_metrics = {}

    for country in combined_df['Country'].unique():
        print(f"\nEvaluating stressed metrics for country: {country}")
        stressed_val_data = combined_df_stressed[
            (combined_df_stressed['Country'] == country) &
            (combined_df_stressed['Date'] >= val_start) &
            (combined_df_stressed['Date'] <= val_end)
        ]
        stressed_blind_test_data = combined_df_stressed[
            (combined_df_stressed['Country'] == country) &
            (combined_df_stressed['Date'] >= blind_test_start) &
            (combined_df_stressed['Date'] <= blind_test_end)
        ]
        stressed_X_val = stressed_val_data.drop(columns=[target_column, 'Date', 'Country'])
        stressed_y_val = stressed_val_data[target_column]
        stressed_X_blind_test = stressed_blind_test_data.drop(columns=[target_column, 'Date', 'Country'])
        stressed_y_blind_test = stressed_blind_test_data[target_column]

        # Preprocess ARIMA-specific dataset
        arima_df = prepare_for_arima_ma(combined_df_stressed[combined_df_stressed['Country'] == country], target_column)

        country_results = {}

        # Default ML models on stressed validation and blind test sets
        for model_name, model in models.items():
            model.fit(stressed_X_val, stressed_y_val)
            stressed_val_preds = model.predict(stressed_X_val)
            stressed_blind_test_preds = model.predict(stressed_X_blind_test)


        country_results[model_name] = {
                'validation': {
                    'MAPE%': mean_absolute_percentage_error(stressed_y_val, stressed_val_preds) * 100,
                    'Accuracy%': 100 - mean_absolute_percentage_error(stressed_y_val, stressed_val_preds) * 100,
                    'Bias%': (np.mean(stressed_val_preds - stressed_y_val) / np.mean(stressed_y_val)) * 100
                },
                'blind_test': {
                    'MAPE%': mean_absolute_percentage_error(stressed_y_blind_test, stressed_blind_test_preds) * 100,
                    'Accuracy%': 100 - mean_absolute_percentage_error(stressed_y_blind_test, stressed_blind_test_preds) * 100,
                    'Bias%': (np.mean(stressed_blind_test_preds - stressed_y_blind_test) / np.mean(stressed_y_blind_test)) * 100
                }
            }

        # Retrained ML models on stressed blind test set only
        for model_name, model in retrained_models.items():
            model.fit(stressed_X_blind_test, stressed_y_blind_test)
            stressed_re_blind_test_preds = model.predict(stressed_X_blind_test)

            country_results[model_name] = {
                'blind_test': {
                    'MAPE%': mean_absolute_percentage_error(stressed_y_blind_test, stressed_re_blind_test_preds) * 100,
                    'Accuracy%': 100 - mean_absolute_percentage_error(stressed_y_blind_test, stressed_re_blind_test_preds) * 100,
                    'Bias%': (np.mean(stressed_re_blind_test_preds - stressed_y_blind_test) / np.mean(stressed_y_blind_test)) * 100
                }
            }

        
        # Use pre-trained ARIMA and MA models
        try:
            _, stressed_arima_forecast, _ = train_arima_model(
                y_combined=arima_df[target_column],
                blind_test_data=stressed_y_blind_test,
                forecast_steps=len(stressed_y_blind_test)
            )

            # Store metrics for ARIMA and MA models
            country_results['ARIMA'] = {
                'blind_test': {
                    'MAPE%': mean_absolute_percentage_error(stressed_y_blind_test, stressed_arima_forecast) * 100,
                    'Accuracy%': 100 - mean_absolute_percentage_error(stressed_y_blind_test, stressed_arima_forecast) * 100,
                    'Bias%': (np.mean(stressed_arima_forecast - stressed_y_blind_test) / np.mean(stressed_y_blind_test)) * 100
                }
            }
        except Exception as e:
            print(f"ARIMA model failed for {country}: {e}")

        try:
            _, stressed_ma_forecast, _ = train_ma_model(
                y_combined=arima_df[target_column],
                blind_test_data=stressed_y_blind_test,
                window=4,
                forecast_steps=len(stressed_y_blind_test)
            )

            country_results['Moving Average'] = {
                'blind_test': {
                    'MAPE%': mean_absolute_percentage_error(stressed_y_blind_test, stressed_ma_forecast) * 100,
                    'Accuracy%': 100 - mean_absolute_percentage_error(stressed_y_blind_test, stressed_ma_forecast) * 100,
                    'Bias%': (np.mean(stressed_ma_forecast - stressed_y_blind_test) / np.mean(stressed_y_blind_test)) * 100
                }
            }
        except Exception as e:
            print(f"Moving Average model failed for {country}: {e}")

        # Add the results for this country to the stressed_metrics dictionary
        stressed_metrics[country] = country_results

    return stressed_metrics


# -------------------------------- Pipeline Model ---------------------------------------------------------------------------------


def full_model_evaluation_pipeline(combined_df, shock_years, shock_quarter, shock_features, shock_magnitude, cycles=3):
    results = {}

    target_column = 'NET Claims Incurred'
    train_start, train_end = "2016-07-01", "2021-12-31"
    val_start, val_end = "2022-01-01", "2023-03-31"
    blind_test_start, blind_test_end = "2023-04-01", "2024-03-31"

    (X_train, y_train), (X_val, y_val), (X_blind_test, y_blind_test) = split_data(combined_df, train_start, train_end, val_start, val_end, blind_test_start, blind_test_end, target_column)

    xgb_model = XGBRegressor()
    lgb_model = LGBMRegressor()
    results['default_xgb_metrics'], xgb_val_preds, xgb_test_preds, xgb_all_preds = train_and_evaluate_model(xgb_model, X_train, y_train, X_val, y_val, X_blind_test, y_blind_test, combined_df, target_column)
    results['default_lgb_metrics'], lgb_val_preds, lgb_test_preds, lgb_all_preds = train_and_evaluate_model(lgb_model, X_train, y_train, X_val, y_val, X_blind_test, y_blind_test, combined_df, target_column)

    # Save all-country, validation, and blind test predictions for default models
    results['ml_predictions'] = {
        'all_countries_predictions': {
            'Default XGBoost': xgb_all_preds,
            'Default LightGBM': lgb_all_preds
        },
        'validation_predictions': {
            'Default XGBoost': xgb_val_preds,
            'Default LightGBM': lgb_val_preds
        },
        'blind_test_predictions': {
            'Default XGBoost': xgb_test_preds,
            'Default LightGBM': lgb_test_preds
        }
    }

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
        'reg_lambda': ('suggest_int', 0, 1)
    }


    best_xgb_params = tune_model(XGBRegressor, X_train, y_train, X_val, y_val, xgb_trial_params)
    best_lgb_params = tune_model(LGBMRegressor, X_train, y_train, X_val, y_val, lgb_trial_params)

    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = pd.concat([y_train, y_val], ignore_index=True)

    re_xgb_model, results['retrained_xgb_metrics'], re_xgb_test_preds, re_xgb_all_preds = retrain_and_evaluate(
        XGBRegressor, best_xgb_params, X_combined, y_combined, X_blind_test, y_blind_test, combined_df, target_column
    )
    re_lgb_model, results['retrained_lgb_metrics'], re_lgb_test_preds, re_lgb_all_preds = retrain_and_evaluate(
        LGBMRegressor, best_lgb_params, X_combined, y_combined, X_blind_test, y_blind_test, combined_df, target_column
    )

    # Add retrained all-country predictions
    results['ml_predictions']['blind_test_predictions']['Retrained XGBoost'] = re_xgb_test_preds
    results['ml_predictions']['blind_test_predictions']['Retrained LightGBM'] = re_lgb_test_preds
    results['ml_predictions']['all_countries_predictions']['Retrained XGBoost'] = re_xgb_all_preds
    results['ml_predictions']['all_countries_predictions']['Retrained LightGBM'] = re_lgb_all_preds


    models = {
        'Default XGBoost': xgb_model,
        'Default LightGBM': lgb_model
    }
    retrained_models = {
        'Retrained XGBoost': re_xgb_model,
        'Retrained LightGBM': re_lgb_model
    }

    # Add ML models to results
    results['ml_models'] = {
        'default_models': models,
        'retrained_models': retrained_models
    }

    # Define model_dict and model_types for backtesting
    model_dict = [models, retrained_models]
    model_types = {
        'Default XGBoost': 'xgb', 
        'Retrained XGBoost': 'xgb', 
        'Default LightGBM': 'lgb', 
        'Retrained LightGBM': 'lgb'
    }

    # Run country-wise evaluation and backtesting
    results['country_metrics'], arima_models, ma_models = evaluate_models_by_country(models, retrained_models, combined_df, target_column, val_start, val_end, blind_test_start, blind_test_end, y_combined)
    results['backtesting_results'] = run_backtest(model_dict, model_types, combined_df=combined_df, target_column=target_column, cycles=cycles)
    
    results['time_series_models'] = {
        'ARIMA': arima_models,
        'Moving Average': ma_models
    }

    results['stressed_metrics'] = apply_stress_to_dataframe(
        combined_df, shock_years, shock_quarter, shock_features, shock_magnitude, 
        models, retrained_models, target_column, val_start, val_end, blind_test_start, blind_test_end
    )


    return results


# -------------------------------- Saving results ---------------------------------------------------------------------------------



def get_or_generate_results(combined_df, cycles=3):
    results_file = 'results.pkl'

    shock_years = [2017, 2020, 2023]
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




#------------------New App with working code------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
# #----------------------------------- Train Default Models -----------------------------------

# def train_default_models(combined_df):    
#     target_column = 'NET Claims Incurred'
#     train_start, train_end =  "2016-07-01", "2021-12-31" 
#     val_start, val_end = "2022-01-01", "2023-03-31"
#     blind_test_start, blind_test_end = "2023-04-01", "2024-03-31" 

#     train_data = combined_df[(combined_df['Date'] >= train_start) & (combined_df['Date'] <= train_end)]
#     val_data = combined_df[(combined_df['Date'] >= val_start) & (combined_df['Date'] <= val_end)]
#     blind_test_data = combined_df[(combined_df['Date'] >= blind_test_start) & (combined_df['Date'] <= blind_test_end)]

#     target_column = 'NET Claims Incurred'
#     X_train = train_data.drop(columns=[target_column, 'Date', 'Country'])
#     y_train = train_data[target_column]

#     X_val = val_data.drop(columns=[target_column, 'Date', 'Country'])
#     y_val = val_data[target_column]

#     X_blind_test = blind_test_data.drop(columns=[target_column, 'Date', 'Country'])
#     y_blind_test = blind_test_data[target_column]

#     xgb_model = XGBRegressor()
#     xgb_model.fit(X_train, y_train)
#     xgb_val_preds = xgb_model.predict(X_val)
#     xgb_blind_test_preds = xgb_model.predict(X_blind_test)
    
#     lgb_model = LGBMRegressor()
#     lgb_model.fit(X_train, y_train)
#     lgb_val_preds = lgb_model.predict(X_val)
#     lgb_blind_test_preds = lgb_model.predict(X_blind_test)
    
#     return xgb_val_preds, xgb_blind_test_preds, lgb_val_preds, lgb_blind_test_preds

# #----------------------------------- Tune and Train Optimized Models -----------------------------------

# def train_optimized_models(combined_df, n_trials=20):
#     target_column = 'NET Claims Incurred'
#     train_start, train_end = "2016-07-01", "2021-12-31"
#     val_start, val_end = "2022-01-01", "2023-03-31"  
#     blind_test_start, blind_test_end = "2023-04-01", "2024-03-31" 

#     train_data = combined_df[(combined_df['Date'] >= train_start) & (combined_df['Date'] <= train_end)]
#     val_data = combined_df[(combined_df['Date'] >= val_start) & (combined_df['Date'] <= val_end)]
#     blind_test_data = combined_df[(combined_df['Date'] >= blind_test_start) & (combined_df['Date'] <= blind_test_end)]

#     X_train = train_data.drop(columns=[target_column, 'Date', 'Country'])
#     y_train = train_data[target_column]
#     X_val = val_data.drop(columns=[target_column, 'Date', 'Country'])
#     y_val = val_data[target_column]
#     X_blind_test = blind_test_data.drop(columns=[target_column, 'Date', 'Country'])
#     y_blind_test = blind_test_data[target_column]

#     def xgb_objective(trial):
#         params = {
#             'n_estimators': trial.suggest_int('n_estimators', 50, 300),
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#             'max_depth': trial.suggest_int('max_depth', 3, 10),
#             'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#             'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
#             'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
#         }
        
#         model = XGBRegressor(**params, objective='reg:squarederror', random_state=42)
#         model.fit(X_train, y_train)
#         val_preds = model.predict(X_val)
#         rmse = sqrt(mean_squared_error(y_val, val_preds))
#         return rmse

#     def lgb_objective(trial):
#         params = {
#             'n_estimators': trial.suggest_int('n_estimators', 50, 300),
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#             'max_depth': trial.suggest_int('max_depth', 3, 10),
#             'num_leaves': trial.suggest_int('num_leaves', 20, 50),
#             'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#             'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
#             'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
#         }
        
#         model = LGBMRegressor(**params, random_state=42)
#         model.fit(X_train, y_train)
#         val_preds = model.predict(X_val)
#         rmse = sqrt(mean_squared_error(y_val, val_preds))
#         return rmse

#     # Run Optuna optimization
#     xgb_study = optuna.create_study(direction='minimize')
#     xgb_study.optimize(xgb_objective, n_trials=20)
#     best_xgb_params = xgb_study.best_params

#     lgb_study = optuna.create_study(direction='minimize')
#     lgb_study.optimize(lgb_objective, n_trials=20)
#     best_lgb_params = lgb_study.best_params

#     X_combined = pd.concat([X_train, X_val])
#     y_combined = pd.concat([y_train, y_val])

#     re_xgb_model = XGBRegressor(**best_xgb_params, objective='reg:squarederror', random_state=42)
#     re_xgb_model.fit(X_combined, y_combined)
#     re_xgb_blind_test_preds = re_xgb_model.predict(X_blind_test)

#     re_lgb_model = LGBMRegressor(**best_lgb_params, random_state=42)
#     re_lgb_model.fit(X_combined, y_combined)
#     re_lgb_blind_test_preds = re_lgb_model.predict(X_blind_test)
    
#     return re_xgb_blind_test_preds, re_lgb_blind_test_preds, best_xgb_params, best_lgb_params


#------------------Simulation App ----------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
# #-------- Function to get XGBoost predictions --------------------------------------------------



# def get_xgboost_predictions(X_combined, y_combined, X_blind_test):
#     re_xgb_model = XGBRegressor(**xgb_best_params)
#     re_xgb_model.fit(X_combined, y_combined)
#     return re_xgb_model.predict(X_blind_test)



# #--------- Function to get LightGBM predictions --------------------------------------------------



# def get_lightgbm_predictions(X_combined, y_combined, X_blind_test):
#     re_lgb_model = LGBMRegressor(**lgb_best_params)
#     re_lgb_model.fit(X_combined, y_combined)
#     return re_lgb_model.predict(X_blind_test)



# #---------- Function to get ARIMA predictions -----------------------------------------------------




# def get_arima_predictions(arima_train_data, arima_test_data, best_pdq, best_seasonal_pdq):
#     # Define target for training
#     arima_y_train = arima_train_data['Claims_Incurred']
    
#     # Fit the final ARIMA model with the manually set best order and seasonal order
#     final_arima_model = sm.tsa.SARIMAX(arima_y_train, 
#                                        order=best_pdq, 
#                                        seasonal_order=best_seasonal_pdq,
#                                        enforce_stationarity=False,  
#                                        enforce_invertibility=False)
    
#     # Fit the model with more iterations and a stricter tolerance
#     final_arima_model_fit = final_arima_model.fit(disp=False, maxiter=1000, tol=1e-6)
    
#     # Forecast for the test period
#     arima_forecast = final_arima_model_fit.forecast(steps=len(arima_test_data))
    
#     return arima_forecast



# #-------- Function to get Moving Average predictions -------------------------------------------------


# def get_moving_average_predictions(ma_y_test, window_size):
#     return ma_y_test.rolling(window=window_size).mean().shift(1).fillna(ma_y_test.mean())



# #----------- Function to get performance metrics -------------------------------------------------



# def calculate_model_metrics(y_true, y_pred):
#     # Convert to numpy arrays
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)

#     # Handle cases where y_true contains zeros to avoid division by zero in MAPE
#     y_true_safe = np.where(y_true == 0, np.finfo(float).eps, y_true)  # Replace 0 with a small epsilon value

#     bias = np.mean(y_pred - y_true)

#     # using the safe version of y_true
#     mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

#     accuracy = 100 - mape

#     return {'Bias': bias, 'MAPE': mape, 'Accuracy': accuracy}



# #----------- Function to get SHAP Analysis -------------------------------------------------------------------------------



# # Ensure your project root and assets folder is correctly targeted
# assets_folder_path = os.path.join(os.getcwd(), 'assets')

# def generate_shap_plot_xgboost(X_combined, y_combined, X_blind_test):
#     re_xgb_model = XGBRegressor(**xgb_best_params)
#     re_xgb_model.fit(X_combined, y_combined)
#     explainer = shap.Explainer(re_xgb_model, X_blind_test)
#     shap_values = explainer(X_blind_test)

#     plt.figure()
#     shap.summary_plot(shap_values, X_blind_test, show=False)  # Prevent it from showing directly
#     save_path_xgb = os.path.join(assets_folder_path, 'shap_summary_xgboost.png')
#     plt.savefig(save_path_xgb)  # Save the plot to assets folder
#     plt.close()  # Close the plot to avoid memory issues

# def generate_shap_plot_lightgbm(X_combined, y_combined, X_blind_test):
#     re_lgb_model = LGBMRegressor(**lgb_best_params)
#     re_lgb_model.fit(X_combined, y_combined)
#     explainer = shap.Explainer(re_lgb_model, X_blind_test)
#     shap_values = explainer(X_blind_test)

#     plt.figure()
#     shap.summary_plot(shap_values, X_blind_test, show=False) 
#     save_path_lgb = os.path.join(assets_folder_path, 'shap_summary_lightgbm.png')
#     plt.savefig(save_path_lgb)  
#     plt.close()  

# # Generate SHAP plots by calling the functions
# generate_shap_plot_xgboost(X_combined, y_combined, X_blind_test)
# generate_shap_plot_lightgbm(X_combined, y_combined, X_blind_test)



# #-------- Function to get feature importance -------------------------------------------------------------



# def get_xgboost_feature_importance(X_combined, y_combined):
#     re_xgb_model = XGBRegressor(**xgb_best_params)
#     re_xgb_model.fit(X_combined, y_combined)
    
#     # Get feature importance and return as a DataFrame
#     re_feature_importance_xgb = pd.DataFrame({
#         'Feature': X_combined.columns,
#         'Importance': re_xgb_model.feature_importances_
#     })

#     re_feature_importance_xgb = re_feature_importance_xgb.sort_values(by='Importance', ascending=False)
#     return re_feature_importance_xgb

# def get_lightgbm_feature_importance(X_combined, y_combined):
#     re_lgb_model = LGBMRegressor(**lgb_best_params)
#     re_lgb_model.fit(X_combined, y_combined)
    
#     # Get feature importance and return as a DataFrame
#     re_feature_importance_lgb = pd.DataFrame({
#         'Feature': X_combined.columns,
#         'Importance': re_lgb_model.feature_importances_
#     })

#     re_feature_importance_lgb = re_feature_importance_lgb.sort_values(by='Importance', ascending=False)
#     return re_feature_importance_lgb



# #------------- Function to get Stress Testing ---------------------------------------------------------



# window_size = 3

# # Prepare data for stress testing
# property_data_feature_selected_s = property_data_feature_selected.drop(['Claims_Incurred', 'Date'], axis=1, errors='ignore')
# property_data_model_s = property_data_model.drop(['Claims_Incurred', 'Date'], axis=1, errors='ignore')

# # Actual data for ML and ts models
# ml_actual_y = property_data_feature_selected['Claims_Incurred']
# ts_actual_y = property_data_model['Claims_Incurred']

# def get_xgboost_predictions_storm():
#     return re_xgb_model.predict(property_data_feature_selected_s)

# def get_lightgbm_predictions_storm():
#     return re_lgb_model.predict(property_data_feature_selected_s)

# def get_arima_predictions_storm():
#     arima_y_train = arima_train_data['Claims_Incurred']
#     final_arima_model = sm.tsa.SARIMAX(arima_y_train, 
#                                        order=best_pdq, 
#                                        seasonal_order=best_seasonal_pdq,
#                                        enforce_stationarity=False,  
#                                        enforce_invertibility=False)
    
#     final_arima_model_fit = final_arima_model.fit(disp=False, maxiter=1000, tol=1e-6)
#     return final_arima_model_fit.forecast(steps=len(property_data_model))

# def get_moving_average_predictions_storm():
#     storm_ma_prediction = ts_actual_y.rolling(window=window_size).mean().shift(1)
#     storm_ma_prediction.fillna(ts_actual_y.mean(), inplace=True)
#     return storm_ma_prediction




# #------------- Function to get Back Testing -------------------------------------------------------------------



# def backtest_with_coc(data, model_type, window=24, test_size=12, num_cycles=3):
#     metrics = {'bias': [], 'accuracy': [], 'mape': []}
    
#     for i in range(num_cycles):
#         train, test = data.iloc[i:i + window], data.iloc[i + window:i + window + test_size]
#         b_X_train = train.drop(['Claims_Incurred', 'Date'], axis=1, errors='ignore')
#         b_y_train = train['Claims_Incurred']
#         b_X_test = test.drop(['Claims_Incurred', 'Date'], axis=1, errors='ignore')
#         b_y_test = test['Claims_Incurred']

#         if model_type == 'xgb':
#             preds = XGBRegressor().fit(b_X_train, b_y_train).predict(b_X_test)
#         elif model_type == 'lgb':
#             preds = LGBMRegressor().fit(b_X_train, b_y_train).predict(b_X_test)
#         elif model_type == 'arima':
#             arima_model = sm.tsa.SARIMAX(b_y_train, order=(3, 2, 3), seasonal_order=(1, 1, 1, 12))
#             arima_fitted = arima_model.fit(disp=False)
#             preds = arima_fitted.forecast(test_size)  # Forecast the test size for ARIMA
#         elif model_type == 'ma':
#             preds = [b_y_train.rolling(window=3).mean().iloc[-1]] * test_size  # Repeat the MA prediction for test size
        
#         bias = np.mean(preds - b_y_test)
#         accuracy = 100 - (mean_absolute_percentage_error(b_y_test, preds) * 100)
#         mape = mean_absolute_percentage_error(b_y_test, preds) * 100

        
#         metrics['bias'].append(bias)
#         metrics['accuracy'].append(accuracy)
#         metrics['mape'].append(mape)

#     return metrics


# def get_xgb_backtest_results(data):
#     return backtest_with_coc(data, 'xgb', window=66, test_size=12, num_cycles=3)

# def get_lgb_backtest_results(data):
#     return backtest_with_coc(data, 'lgb', window=66, test_size=12, num_cycles=3)

# def get_arima_backtest_results(data):
#     return backtest_with_coc(data, 'arima', window=66, test_size=12, num_cycles=3)

# def get_ma_backtest_results(data):
#     return backtest_with_coc(data, 'ma', window=66, test_size=12, num_cycles=3)




# #------------- Function to get Business Solution -------------------------------------------------------------------




# def generate_forecast_tables(final_arima_model_fit):
#     # ARIMA Forecast (Nov 2024 - Oct 2025)
#     forecast_dates = pd.date_range(start='2024-11-01', periods=12, freq='ME')
#     arima_future_predictions = final_arima_model_fit.forecast(steps=12)

#     arima_forecast_table = pd.DataFrame({
#         'LOB': 'Property',
#         'Date': forecast_dates.strftime('%Y-%b'),  # Format YYYYMon
#         'Prediction': arima_future_predictions,
#         'Model': 'ARIMA'
#     })

#     arima_forecast_table.reset_index(drop=True, inplace=True)

#     # Since ARIMA is now the only model being considered, the best prediction is always ARIMA
#     final_forecast_table = arima_forecast_table[['LOB', 'Date', 'Prediction', 'Model']].copy()

#     # Modify the 'LOB' column to only show 'Property' in the middle row
#     middle_row = len(final_forecast_table) // 2
#     final_forecast_table['LOB'] = [''] * len(final_forecast_table)  # Empty the LOB column
#     final_forecast_table.loc[middle_row, 'LOB'] = 'Property'  # Set 'Property' in the middle row
    
#     return final_forecast_table
