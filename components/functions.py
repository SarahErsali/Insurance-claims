import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import shap
from math import sqrt
import optuna
#from components.data import X_train, y_train, X_val, y_val, X_blind_test, property_data_model, property_data_feature_selected
import matplotlib
matplotlib.use('Agg')  # Switch to a non-interactive backend
import matplotlib.pyplot as plt
import os
#from components.data import convert_quarter_to_date, load_and_process_uploaded_data



#----------------------------------- Train Default Models -----------------------------------

def train_default_models(combined_df):    
    target_column = 'NET Claims Incurred'
    train_start, train_end = "2016-07-01", "2020-12-31"
    blind_test_start, blind_test_end = "2023-01-01", "2024-03-31"

    train_data = combined_df[(combined_df['Date'] >= train_start) & (combined_df['Date'] <= train_end)]
    blind_test_data = combined_df[(combined_df['Date'] >= blind_test_start) & (combined_df['Date'] <= blind_test_end)]

    X_train = train_data.drop(columns=[target_column, 'Date'])
    y_train = train_data[target_column]
    X_blind_test = blind_test_data.drop(columns=[target_column, 'Date'])

    xgb_model = XGBRegressor()
    xgb_model.fit(X_train, y_train)
    xgb_blind_test_preds = xgb_model.predict(X_blind_test)
    
    lgb_model = LGBMRegressor()
    lgb_model.fit(X_train, y_train)
    lgb_blind_test_preds = lgb_model.predict(X_blind_test)
    
    return xgb_blind_test_preds, lgb_blind_test_preds

#----------------------------------- Tune and Train Optimized Models -----------------------------------

def train_optimized_models(combined_df, n_trials=20):
    target_column = 'NET Claims Incurred'
    train_start, train_end = "2016-07-01", "2020-12-31"
    val_start, val_end = "2021-01-01", "2022-12-31"
    blind_test_start, blind_test_end = "2023-01-01", "2024-03-31"

    train_data = combined_df[(combined_df['Date'] >= train_start) & (combined_df['Date'] <= train_end)]
    val_data = combined_df[(combined_df['Date'] >= val_start) & (combined_df['Date'] <= val_end)]
    blind_test_data = combined_df[(combined_df['Date'] >= blind_test_start) & (combined_df['Date'] <= blind_test_end)]

    X_train = train_data.drop(columns=[target_column, 'Date'])
    y_train = train_data[target_column]
    X_val = val_data.drop(columns=[target_column, 'Date'])
    y_val = val_data[target_column]
    X_blind_test = blind_test_data.drop(columns=[target_column, 'Date'])

    def xgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        }
        
        model = XGBRegressor(**params, objective='reg:squarederror', random_state=42)
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        rmse = sqrt(mean_squared_error(y_val, val_preds))
        return rmse

    def lgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 50),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        }
        
        model = LGBMRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        rmse = sqrt(mean_squared_error(y_val, val_preds))
        return rmse

    # Run Optuna optimization
    xgb_study = optuna.create_study(direction='minimize')
    xgb_study.optimize(xgb_objective, n_trials=n_trials)
    best_xgb_params = xgb_study.best_params

    lgb_study = optuna.create_study(direction='minimize')
    lgb_study.optimize(lgb_objective, n_trials=n_trials)
    best_lgb_params = lgb_study.best_params

    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])

    re_xgb_model = XGBRegressor(**best_xgb_params, objective='reg:squarederror', random_state=42)
    re_xgb_model.fit(X_combined, y_combined)
    re_xgb_blind_test_preds = re_xgb_model.predict(X_blind_test)

    re_lgb_model = LGBMRegressor(**best_lgb_params, random_state=42)
    re_lgb_model.fit(X_combined, y_combined)
    re_lgb_blind_test_preds = re_lgb_model.predict(X_blind_test)
    
    return re_xgb_blind_test_preds, re_lgb_blind_test_preds, best_xgb_params, best_lgb_params










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
