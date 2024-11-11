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
#from components.data import convert_quarter_to_date, load_and_process_uploaded_data





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

    return (X_train, y_train), (X_val, y_val), (X_blind_test, y_blind_test)


# ---------------------------------- Default Model Developement -----------------------------------------------------



def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_blind_test=None, y_blind_test=None):
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
    return metrics, val_preds, blind_test_preds


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
        
            print("Trial parameters generated:", params)
        
            # Create and train model with these parameters
            model = model_class(**params)
            model.fit(X_train, y_train)
            val_preds = model.predict(X_val)
            return sqrt(mean_squared_error(y_val, val_preds))
        
        except TypeError as e:
            print(f"Error in parameter setting: {e}")
            raise e  # Re-raise to capture Optuna logging

    # Check the initial trial_params input
    print("Trial parameters received:", trial_params)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    return study.best_params



# ----------------------------------- Retraining on Blind Test Set -----------------------------------------


def retrain_and_evaluate(model_class, best_params, X_combined, y_combined, X_blind_test, y_blind_test):
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
    return model, metrics, test_preds



# ----------------------------------- Developing Time Series Model -----------------------------------------


def train_arima_model(y_train, forecast_steps, max_p=5, max_q=5, max_d=2):
    arima_params_model = auto_arima(
        y_train, seasonal=False, stepwise=True, trace=False, 
        error_action='ignore', max_p=max_p, max_q=max_q, max_d=max_d
    )
    best_params = arima_params_model.order
    arima_model = ARIMA(y_train, order=best_params)
    arima_model_fitted = arima_model.fit()
    
    forecast = arima_model_fitted.forecast(steps=forecast_steps).values.flatten()
    
    return arima_model_fitted, forecast


def train_ma_model(y_train, window, forecast_steps):
    ma_model = y_train.rolling(window=window).mean()
    
    forecast = ma_model.dropna().iloc[-forecast_steps:].values.flatten()
    
    return ma_model, forecast




# ----------------------------------- Model Evaluation on Each Country -----------------------------------------

def evaluate_models_by_country(models, retrained_models, combined_df, target_column, val_start, val_end, blind_test_start, blind_test_end, y_combined):
    countries = combined_df['Country'].unique()
    results = {}
    arima_models = {}
    ma_models = {}


    for country in countries:
        country_data = combined_df[combined_df['Country'] == country]

        _, (X_val_country, y_val_country), (X_blind_test_country, y_blind_test_country) = split_data(
            country_data, val_start, val_end, blind_test_start, blind_test_end, target_column
        )
        country_results = {}

        for model_name, model in models.items():
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

            country_results[f"{model_name} Default"] = {
                'metrics': metrics,
                'validation_predictions': val_preds,
                'blind_test_predictions': blind_test_preds
            }

        for model_name, model in retrained_models.items():
            blind_test_preds = model.predict(X_blind_test_country)

            metrics = {
                'blind_test': {
                    'MAPE%': mean_absolute_percentage_error(y_blind_test_country, blind_test_preds) * 100,
                    'Accuracy%': 100 - mean_absolute_percentage_error(y_blind_test_country, blind_test_preds) * 100,
                    'Bias%': (np.mean(blind_test_preds - y_blind_test_country) / np.mean(y_blind_test_country)) * 100
                }
            }

            country_results[f"{model_name} Retrained"] = {
                'metrics': metrics,
                'blind_test_predictions': blind_test_preds
            }

        arima_model, arima_forecast = train_arima_model(y_train=y_combined[combined_df['Country'] == country], forecast_steps=len(y_blind_test_country))
        ma_model, ma_forecast = train_ma_model(y_train=y_combined[combined_df['Country'] == country], window=4, forecast_steps=len(y_blind_test_country))

        # Store each country's ARIMA and MA models in their respective dictionaries
        arima_models[country] = arima_model
        ma_models[country] = ma_model

        # Store ARIMA and MA metrics
        arima_metrics = {
            'blind_test': {
                'MAPE%': mean_absolute_percentage_error(y_blind_test_country, arima_forecast) * 100,
                'Accuracy%': 100 - mean_absolute_percentage_error(y_blind_test_country, arima_forecast) * 100,
                'Bias%': (np.mean(arima_forecast - y_blind_test_country) / np.mean(y_blind_test_country)) * 100
            }
        }

        ma_metrics = {
            'blind_test': {
                'MAPE%': mean_absolute_percentage_error(y_blind_test_country, ma_forecast) * 100,
                'Accuracy%': 100 - mean_absolute_percentage_error(y_blind_test_country, ma_forecast) * 100,
                'Bias%': (np.mean(ma_forecast - y_blind_test_country) / np.mean(y_blind_test_country)) * 100
            }
        }

        country_results['ARIMA'] = {
            'metrics': arima_metrics,
            'blind_test_predictions': arima_forecast
        }
        country_results['Moving Average'] = {
            'metrics': ma_metrics,
            'blind_test_predictions': ma_forecast
        }

        arima_models[country] = arima_model
        ma_models[country] = ma_model
        results[country] = country_results

    return results, arima_models, ma_models




# ---------------------------------- Back Testing ----------------------------------------------------------------

# model_dicts is models
# model_types is the dictionary defining each model's type

def run_backtest(model_dicts, model_types, combined_df, target_column, cycles):
    # Combine all models and types into one dictionary
    all_models = {}
    all_models.update(*model_dicts)  # Combine all models into one dictionary

    countries = combined_df['Country'].unique()
    backtesting_results = {}

    for country in countries:
        country_data = combined_df[combined_df['Country'] == country]
        
        model_results = {}
        for model_name, model in all_models.items():
            model_type = model_types[model_name]  # Get the model type directly from model_types dictionary
            backtest_metrics = backtest_model(country_data, model, model_type, cycles=cycles)  # Pass model_type directly to backtest_model
            model_results[model_name] = backtest_metrics
        
        backtesting_results[country] = model_results

    return backtesting_results





def backtest_model(country_data, model, model_type, window=24, test_size=12, cycles=3):
    cycle_metrics = {'bias': [], 'accuracy': [], 'mape': [], 'coc': []}
    target_column = 'NET Claims Incurred'
    prev_cycle_preds = None

    # Pre-set the prediction approach based on model type
    is_ml_model = model_type in ['xgb', 'lgb']
    is_arima_model = model_type == 'arima'
    is_ma_model = model_type == 'ma'

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
            arima_cycle_model = ARIMA(b_y_train, order=model.order).fit()
            preds = arima_cycle_model.forecast(steps=len(b_y_test)).values
        elif is_ma_model:
            preds = b_y_train.rolling(window=4).mean().iloc[-len(b_y_test):].dropna().values


        if len(preds) == len(b_y_test):
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

        country_results = {}

        # Default ML models on stressed validation and blind test sets
        for model_name, model in models.items():
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
            stressed_re_blind_test_preds = model.predict(stressed_X_blind_test)

            country_results[f"{model_name} Retrained"] = {
                'blind_test': {
                    'MAPE%': mean_absolute_percentage_error(stressed_y_blind_test, stressed_re_blind_test_preds) * 100,
                    'Accuracy%': 100 - mean_absolute_percentage_error(stressed_y_blind_test, stressed_re_blind_test_preds) * 100,
                    'Bias%': (np.mean(stressed_re_blind_test_preds - stressed_y_blind_test) / np.mean(stressed_y_blind_test)) * 100
                }
            }


        # Train and evaluate ARIMA and MA models on stressed data for the country
        arima_model, stressed_arima_forecast = train_arima_model(
            y_train=stressed_y_val,  # Train ARIMA on stressed validation data
            forecast_steps=len(stressed_y_blind_test)
        )
        ma_model, stressed_ma_forecast = train_ma_model(
            y_train=stressed_y_val,
            window=4,
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
        country_results['Moving Average'] = {
            'blind_test': {
                'MAPE%': mean_absolute_percentage_error(stressed_y_blind_test, stressed_ma_forecast) * 100,
                'Accuracy%': 100 - mean_absolute_percentage_error(stressed_y_blind_test, stressed_ma_forecast) * 100,
                'Bias%': (np.mean(stressed_ma_forecast - stressed_y_blind_test) / np.mean(stressed_y_blind_test)) * 100
            }
        }

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
    results['default_xgb_metrics'], xgb_val_preds, xgb_test_preds = train_and_evaluate_model(xgb_model, X_train, y_train, X_val, y_val, X_blind_test, y_blind_test)
    results['default_lgb_metrics'], lgb_val_preds, lgb_test_preds = train_and_evaluate_model(lgb_model, X_train, y_train, X_val, y_val, X_blind_test, y_blind_test)

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

    # print("xgb_trial_params:", xgb_trial_params)
    # print("lgb_trial_params:", lgb_trial_params)

    best_xgb_params = tune_model(XGBRegressor, X_train, y_train, X_val, y_val, xgb_trial_params)
    best_lgb_params = tune_model(LGBMRegressor, X_train, y_train, X_val, y_val, lgb_trial_params)

    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = pd.concat([y_train, y_val], ignore_index=True)

    re_xgb_model, results['retrained_xgb_metrics'], re_xgb_test_preds = retrain_and_evaluate(
        XGBRegressor, best_xgb_params, X_combined, y_combined, X_blind_test, y_blind_test
    )
    re_lgb_model, results['retrained_lgb_metrics'], re_lgb_test_preds = retrain_and_evaluate(
        LGBMRegressor, best_lgb_params, X_combined, y_combined, X_blind_test, y_blind_test
    )

    models = {
        'Default XGBoost': xgb_model,
        'Default LightGBM': lgb_model
    }
    retrained_models = {
        'Retrained XGBoost': re_xgb_model,
        'Retrained LightGBM': re_lgb_model
    }

    # Define model_dict and model_types for backtesting
    model_dict = {**models, **retrained_models}
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

    # Define the shock parameters
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
