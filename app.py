import pandas as pd
import numpy as np
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import statsmodels.api as sm
from components.tabs.home import render_home
from components.tabs.tab2 import render_tab2
from components.tabs.tab1 import render_tab1
from components.tabs.tab3 import render_tab3
from components.tabs.tab4 import render_tab4
from components.tabs.tab5 import render_tab5
import components.data as data
# from components.functions import (
#     get_xgboost_predictions,
#     get_lightgbm_predictions,
#     get_arima_predictions,
#     get_moving_average_predictions,
#     calculate_model_metrics,
#     arima_test_data,
#     ma_y_test,
#     arima_train_data,
#     get_xgboost_feature_importance,
#     get_lightgbm_feature_importance,
#     property_data_model,
#     get_xgboost_predictions_storm,
#     get_lightgbm_predictions_storm,
#     get_arima_predictions_storm,
#     get_moving_average_predictions_storm,
#     ts_actual_y,
#     get_xgb_backtest_results,
#     get_lgb_backtest_results,
#     get_arima_backtest_results,
#     get_ma_backtest_results,
#     generate_forecast_tables
# )


#--------------------- Initialize the app---------------------


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Insurance Consultant Service"

# Initialize a dictionary to store uploaded data
uploaded_data = {}

# Stores names of all uploaded files
uploaded_files_list = []

# Define the layout of the app
app.layout = html.Div([
    html.Header([
        html.H1("BaNex Consulting Insurance Service", style={'textAlign': 'center', 'fontSize': '48px', 'marginTop': '10px'}),
    ], style={'backgroundColor': '#f0f0f0', 'padding': '50px'}),

    # Navigation tabs
    dcc.Tabs(id='tabs-example', value='home', children=[
        dcc.Tab(label='Business Objectives', value='home'),
        dcc.Tab(label='Data Overview', value='tab-1'),
        dcc.Tab(label='Exploratory Data Analysis', value='tab-2'),
        dcc.Tab(label='Model Performance', value='tab-3'),
        dcc.Tab(label='Model Robustness', value='tab-4'),
        dcc.Tab(label='Business Solution', value='tab-5'),
    ]),

    # Content section that changes with each tab
    html.Div(id='tabs-content', style={'textAlign': 'center', 'padding': '0px', 'height': '50vh'})
])

#----------- Callback to update the page content based on the selected tab ------------------

@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs-example', 'value')
)
def render_content(tab):
    if tab == 'home':
        return render_home()
    elif tab == 'tab-1':
        return render_tab1()
    elif tab == 'tab-2':
        return render_tab2()
    elif tab == 'tab-3':
        return render_tab3()
    elif tab == 'tab-4':
        return render_tab4()
    elif tab == 'tab-5':
        return render_tab5()

#------------ Callback for handling file upload ------------------

@app.callback(
    Output('upload-status', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def update_upload(contents, filenames):
    global uploaded_data, uploaded_files_list
    status_messages = []

    if contents is not None:
        for content, filename in zip(contents, filenames):
            uploaded_data = data.load_uploaded_data(content, filename, uploaded_data)
            message = f"File '{filename}' uploaded and ready to be cleaned."
            uploaded_files_list.append(message)  # Add message to list
            status_messages = uploaded_files_list  # Update the display messages list
        print("Files uploaded and stored in `uploaded_data`:", list(uploaded_data.keys()))  # Debugging output

    return html.Ul([html.Li(msg) for msg in status_messages])



#------------ Callback for cleaning uploaded files ------------------



@app.callback(
    Output('clean-status', 'children'),
    Input('clean-data-button', 'n_clicks'),
    prevent_initial_call=True
)
def clean_data(n_clicks):
    global uploaded_data
    if uploaded_data:
        print("Starting data cleaning...")  # Debugging output

        # Clean all uploaded datasets and update `uploaded_data` with cleaned data
        cleaned_data = data.clean_uploaded_dataframes(uploaded_data)  # Store the cleaned data
        uploaded_data = cleaned_data  # Update the global `uploaded_data` with cleaned data
        print("Data cleaning complete. Current datasets:", list(uploaded_data.keys()))  # Debugging output

        return "Data cleaning is complete. Cleaned data is now ready for feature engineering."
    else:
        print("No data found for cleaning.")  # Debugging output

        return "No data available to clean. Please upload files first."
    


#------------ Callback for feature engineering ------------------

#------------ Callback for feature engineering ------------------

@app.callback(
    Output('feature-engineering-status', 'children'),
    Input('feature-engineering-button', 'n_clicks'),
    prevent_initial_call=True
)
def run_feature_engineering(n_clicks):
    global uploaded_data
    if uploaded_data:
        try:
            print("Starting feature engineering...")  # Debugging output
            # Process feature engineering on all uploaded datasets
            processed_data = data.combine_and_process_data(
                uploaded_data,
                max_lags=8,
                encoding='label'  # Default encoding; could allow user selection if needed
            )
            uploaded_data = processed_data  # Update `uploaded_data` with processed data
            print("Feature engineering complete. Processed data:", list(uploaded_data.keys()))  # Debugging output
            return "Feature engineering is complete. You may now proceed with data exploration and model building."
        except Exception as e:
            print("Error during feature engineering:", str(e))
            return f"Feature engineering failed: {str(e)}"
    else:
        print("No data found for feature engineering.")  # Debugging output
        return "No data available for feature engineering. Please upload and clean files first."



# #--------------- Callback for Model Predictions Plot----------------


# @app.callback(
#     Output('model-comparison-graph', 'figure'),
#     Input('model-dropdown-prediction', 'value')
# )
# def update_model_predictions(models_selected):
#     fig = go.Figure()

#     for model in models_selected:
#         if model == 'xgboost':
#             preds = get_xgboost_predictions(X_combined, y_combined, X_blind_test)
#             preds = pd.Series(preds, index=y_blind_test.reset_index(drop=True))
#             fig.add_trace(go.Scatter(x=y_blind_test.index, y=preds, mode='lines', name='XGBoost', line=dict(color='purple')))

#         if model == 'lightgbm':
#             preds = get_lightgbm_predictions(X_combined, y_combined, X_blind_test)
#             preds = pd.Series(preds, index=y_blind_test.reset_index(drop=True))
#             fig.add_trace(go.Scatter(x=y_blind_test.index, y=preds, mode='lines', name='LightGBM', line=dict(color='blue')))

#         if model == 'arima':
#             preds = get_arima_predictions(arima_train_data, arima_test_data, (3, 2, 3), (1, 1, 1, 12))
#             preds = pd.Series(preds, index=arima_test_data.index)
#             fig.add_trace(go.Scatter(x=y_blind_test.index, y=preds, mode='lines', name='ARIMA', line=dict(color='green')))

#         if model == 'moving_average':
#             preds = get_moving_average_predictions(ma_y_test, window_size=3)
#             preds = pd.Series(preds, index=ma_y_test.index)
#             fig.add_trace(go.Scatter(x=y_blind_test.index, y=preds, mode='lines', name='Moving Average', line=dict(color='red')))

#     # Add the actual values to the original graph
#     fig.add_trace(go.Scatter(x=y_blind_test.index, y=y_blind_test, mode='lines', name='Actual', line=dict(color='black', dash='dot')))
#     fig.update_layout(xaxis_title='Date', yaxis_title='Claims Incurred', xaxis_showgrid=False, yaxis_showgrid=False)

#     return fig


# #-------------- Callback for Metrics Bar Chart --------------


# @app.callback(
#     [Output('model-bias-chart', 'figure'),
#      Output('model-accuracy-chart', 'figure'),
#      Output('model-mape-chart', 'figure')],
#     Input('model-dropdown-metrics', 'value')
# )
# def update_metrics_chart(models_selected):
#     metrics_fig = go.Figure()
#     metrics = {'Bias': [], 'Accuracy': [], 'MAPE': []}
#     model_names = []


#     # Fetch metrics for each selected model
#     for model in models_selected:
#         if model == 'xgboost':
#             preds = get_xgboost_predictions(X_combined, y_combined, X_blind_test)
#             preds = pd.Series(preds, index=y_blind_test.reset_index(drop=True))
#             model_metrics = calculate_model_metrics(y_blind_test, preds)
#             model_names.append('XGBoost')
#             metrics['Bias'].append(model_metrics['Bias'] / 1000)
#             metrics['Accuracy'].append(model_metrics['Accuracy'])
#             metrics['MAPE'].append(model_metrics['MAPE'])

#         if model == 'lightgbm':
#             preds = get_lightgbm_predictions(X_combined, y_combined, X_blind_test)
#             preds = pd.Series(preds, index=y_blind_test.reset_index(drop=True))
#             model_metrics = calculate_model_metrics(y_blind_test, preds)
#             model_names.append('LightGBM')
#             metrics['Bias'].append(model_metrics['Bias'] / 1000)
#             metrics['Accuracy'].append(model_metrics['Accuracy'])
#             metrics['MAPE'].append(model_metrics['MAPE'])

#         if model == 'arima':
#             preds = get_arima_predictions(arima_train_data, arima_test_data, (3, 2, 3), (1, 1, 1, 12))
#             preds = pd.Series(preds, index=arima_test_data.index)
#             model_metrics = calculate_model_metrics(y_blind_test, preds)
#             model_names.append('ARIMA')
#             metrics['Bias'].append(model_metrics['Bias'] / 1000)
#             metrics['Accuracy'].append(model_metrics['Accuracy'])
#             metrics['MAPE'].append(model_metrics['MAPE'])

#         if model == 'moving_average':
#             preds = get_moving_average_predictions(ma_y_test, window_size=3)
#             preds = pd.Series(preds, index=ma_y_test.index)
#             model_metrics = calculate_model_metrics(y_blind_test, preds)
#             model_names.append('Moving Average')
#             metrics['Bias'].append(model_metrics['Bias'] / 1000)
#             metrics['Accuracy'].append(model_metrics['Accuracy'])
#             metrics['MAPE'].append(model_metrics['MAPE'])

    
#     # Convert NaNs to 0 and ensure all values are floats
#     for key in metrics:
#         metrics[key] = [0.0 if np.isnan(value) else float(value) for value in metrics[key]]
    

#     # Create bar charts for each metric
#     bias_fig = go.Figure([go.Bar(x=model_names, y=metrics['Bias'], name='Bias', marker_color='orange')])
#     accuracy_fig = go.Figure([go.Bar(x=model_names, y=metrics['Accuracy'], name='Accuracy', marker_color='green')])
#     mape_fig = go.Figure([go.Bar(x=model_names, y=metrics['MAPE'], name='MAPE', marker_color='blue')])

#     # Layout for each chart
#     bias_fig.update_layout(
#         xaxis_title='Models', 
#         yaxis_title='Bias', 
#         xaxis_showgrid=False, 
#         yaxis_showgrid=False
#     )
#     accuracy_fig.update_layout(
#         xaxis_title='Models', 
#         yaxis_title='Accuracy', 
#         xaxis_showgrid=False, 
#         yaxis_showgrid=False, 
#         yaxis=dict(range=[0, 100])
#     )
#     mape_fig.update_layout(
#         xaxis_title='Models', 
#         yaxis_title='MAPE', 
#         xaxis_showgrid=False, 
#         yaxis_showgrid=False
#     )

#     return bias_fig, accuracy_fig, mape_fig




# #------------- Callback for Feature Importance Bar Chart -------------------



# @app.callback(
#     [Output('xgboost-feature-importance-bar-chart', 'figure'),
#      Output('lightgbm-feature-importance-bar-chart', 'figure')],
#     Input('tabs-example', 'value')
# )
# def update_feature_importance_chart(_):
#     feature_importance_df_xgboost = get_xgboost_feature_importance(X_combined, y_combined)
#     xgboost_fig = go.Figure([go.Bar(
#         x=feature_importance_df_xgboost['Feature'],
#         y=feature_importance_df_xgboost['Importance'],
#         marker_color='orange'
#     )])

#     xgboost_fig.update_layout(
#         xaxis_title='Features',
#         yaxis_title='Importance',
#         xaxis_tickangle=-45,
#         height=500,
#         xaxis_showgrid=False,
#         yaxis_showgrid=False,
#         xaxis={'zeroline': False},
#         yaxis={'zeroline': False}
#     )

#     feature_importance_df_lightgbm = get_lightgbm_feature_importance(X_combined, y_combined)
#     lightgbm_fig = go.Figure([go.Bar(
#         x=feature_importance_df_lightgbm['Feature'],
#         y=feature_importance_df_lightgbm['Importance'],
#         marker_color='blue'
#     )])

#     lightgbm_fig.update_layout(
#         xaxis_title='Features',
#         yaxis_title='Importance',
#         xaxis_tickangle=-45,
#         height=500,
#         xaxis_showgrid=False,
#         yaxis_showgrid=False,
#         xaxis={'zeroline': False},
#         yaxis={'zeroline': False}
#     )

#     return xgboost_fig, lightgbm_fig



# #---------- Callback for Stress Testing Plot -----------



# @app.callback(
#     Output('storm-testing-graph', 'figure'),
#     Input('model-dropdown-storm', 'value')
# )
# def update_storm_testing(models_selected):
#     fig = go.Figure()

#     # Ensure the x-axis uses proper date values
#     actual_dates = property_data_model['Date'] 

#     for model in models_selected:
#         if model == 'xgboost':            
#             xgb_preds_storm = get_xgboost_predictions_storm()
            
#             if len(xgb_preds_storm) > 0:
#                 fig.add_trace(go.Scatter(
#                     x=actual_dates,  
#                     y=xgb_preds_storm, 
#                     mode='lines', 
#                     name='XGBoost', 
#                     line=dict(color='purple')
#                 ))

#         if model == 'lightgbm':            
#             lgb_preds_storm = get_lightgbm_predictions_storm()
            
#             if len(lgb_preds_storm) > 0:
#                 fig.add_trace(go.Scatter(
#                     x=actual_dates, 
#                     y=lgb_preds_storm, 
#                     mode='lines', 
#                     name='LightGBM', 
#                     line=dict(color='blue')
#                 ))

#         if model == 'arima':            
#             arima_preds_storm = get_arima_predictions_storm()
            
#             if len(arima_preds_storm) > 0:
#                 fig.add_trace(go.Scatter(
#                     x=actual_dates,  
#                     y=arima_preds_storm, 
#                     mode='lines', 
#                     name='ARIMA', 
#                     line=dict(color='green')
#                 ))

#         if model == 'moving_average':            
#             ma_preds_storm = get_moving_average_predictions_storm()
            
#             if len(ma_preds_storm) > 0:
#                 fig.add_trace(go.Scatter(
#                     x=actual_dates,  
#                     y=ma_preds_storm, 
#                     mode='lines', 
#                     name='Moving Average', 
#                     line=dict(color='red')
#                 ))

#     # Add actual values across the period
#     fig.add_trace(go.Scatter(
#         x=actual_dates,  
#         y=ts_actual_y, 
#         mode='lines', 
#         name='Actual', 
#         line=dict(color='black', dash='dot')
#     ))

#     fig.update_layout(
#         xaxis_title='Date', 
#         yaxis_title='Claims Incurred During Storms', 
#         xaxis_showgrid=False, 
#         yaxis_showgrid=False,
#         xaxis=dict(type='date')  # Ensure the x-axis is treated as dates
#     )

#     return fig


# #---------- Callback for Backtesting Results -------------


# @app.callback(
#     [Output('backtest-bias-chart', 'figure'),
#      Output('backtest-accuracy-chart', 'figure'),
#      Output('backtest-mape-chart', 'figure')],
#     Input('model-dropdown-backtest', 'value')
# )
# def update_backtest_charts(models_selected):
#     # Initialize a dictionary to store metrics for each selected model
#     metrics = {'bias': {}, 'accuracy': {}, 'mape': {}}

#     # Mapping from dropdown values to the display names used in the metrics dictionary
#     model_name_mapping = {
#         'xgboost': 'XGBoost',
#         'lightgbm': 'LightGBM',
#         'arima': 'ARIMA',
#         'moving_average': 'Moving Average'
#     }

#     # Color scheme for different models (distinguishable shades)
#     color_mapping = {
#         'xgboost': {'bias': 'orange', 'accuracy': 'lightgreen', 'mape': 'lightblue'},
#         'lightgbm': {'bias': 'darkorange', 'accuracy': 'green', 'mape': 'blue'},
#         'arima': {'bias': 'gold', 'accuracy': 'darkgreen', 'mape': 'navy'},
#         'moving_average': {'bias': 'chocolate', 'accuracy': 'forestgreen', 'mape': 'royalblue'}
#     }

#     # Loop through the selected models and retrieve backtest results for each one
#     for model in models_selected:
#         if model == 'xgboost':
#             xgb_metrics = get_xgb_backtest_results(property_data_feature_selected)
#             metrics['bias']['XGBoost'] = xgb_metrics['bias']
#             metrics['accuracy']['XGBoost'] = xgb_metrics['accuracy']
#             metrics['mape']['XGBoost'] = xgb_metrics['mape']

#         elif model == 'lightgbm':
#             lgb_metrics = get_lgb_backtest_results(property_data_feature_selected)
#             metrics['bias']['LightGBM'] = lgb_metrics['bias']
#             metrics['accuracy']['LightGBM'] = lgb_metrics['accuracy']
#             metrics['mape']['LightGBM'] = lgb_metrics['mape']

#         elif model == 'arima':
#             arima_metrics = get_arima_backtest_results(property_data_model[['Date', 'Claims_Incurred']])
#             metrics['bias']['ARIMA'] = arima_metrics['bias']
#             metrics['accuracy']['ARIMA'] = arima_metrics['accuracy']
#             metrics['mape']['ARIMA'] = arima_metrics['mape']

#         elif model == 'moving_average':
#             ma_metrics = get_ma_backtest_results(property_data_model[['Date', 'Claims_Incurred']])
#             metrics['bias']['Moving Average'] = ma_metrics['bias']
#             metrics['accuracy']['Moving Average'] = ma_metrics['accuracy']
#             metrics['mape']['Moving Average'] = ma_metrics['mape']

#     cycles = ['Cycle 1', 'Cycle 2', 'Cycle 3']

#     bias_fig = go.Figure()
#     accuracy_fig = go.Figure()
#     mape_fig = go.Figure()

#     # Add traces to each figure based on selected models
#     for model in models_selected:
#         model_display_name = model_name_mapping[model]
        
#         if model_display_name in metrics['bias']:
#             # Use different colors for each model
#             bias_color = color_mapping[model]['bias']
#             accuracy_color = color_mapping[model]['accuracy']
#             mape_color = color_mapping[model]['mape']

#             bias_fig.add_trace(go.Bar(x=cycles, y=metrics['bias'][model_display_name], name=f'{model_display_name}', marker_color=bias_color))
#             accuracy_fig.add_trace(go.Bar(x=cycles, y=metrics['accuracy'][model_display_name], name=f'{model_display_name}', marker_color=accuracy_color))
#             mape_fig.add_trace(go.Bar(x=cycles, y=metrics['mape'][model_display_name], name=f'{model_display_name}', marker_color=mape_color))

#     bias_fig.update_layout(
#         bargap=0.15, 
#         bargroupgap=0.2,
#         xaxis=dict(showgrid=False), 
#         yaxis=dict(showgrid=False)  
#     )
#     accuracy_fig.update_layout(
#         bargap=0.15, 
#         bargroupgap=0.2,
#         xaxis=dict(showgrid=False),  
#         yaxis=dict(showgrid=False)   
#     )
#     mape_fig.update_layout(
#         bargap=0.15, 
#         bargroupgap=0.2,
#         xaxis=dict(showgrid=False), 
#         yaxis=dict(showgrid=False)   
#     )

#     return bias_fig, accuracy_fig, mape_fig



# #---------- Callback for Business Solution -------------



# @app.callback(
#     Output('future-prediction-table', 'data'),
#     Input('tabs-example', 'value')
# )
# def update_future_prediction_table(tab):
#     if tab == 'tab-5':
#         arima_y_train = arima_train_data['Claims_Incurred']
#         best_pdq = (3, 2, 3)  
#         best_seasonal_pdq = (1, 1, 1, 12) 
#         final_arima_model = sm.tsa.SARIMAX(arima_y_train, order=best_pdq, seasonal_order=best_seasonal_pdq, enforce_stationarity=False, enforce_invertibility=False)
#         final_arima_model_fit = final_arima_model.fit(disp=False)

#         # Call the function to generate the forecast table using the fitted ARIMA model
#         final_forecast_table = generate_forecast_tables(final_arima_model_fit)

#         # Format the 'Prediction' column to scientific notation with 5 decimals
#         final_forecast_table['Prediction'] = final_forecast_table['Prediction'].apply(lambda x:f'{x/1e6:.3f}' if x >= 1e6 else f'{x:.3f}')

        
#         # Convert the DataFrame to a list of dictionaries to use in Dash DataTable
#         table_data = final_forecast_table.to_dict('records')

        
#         return table_data
    
#     return []  # Return an empty table if the tab is not 'tab-5'




# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
