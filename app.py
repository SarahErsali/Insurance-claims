import pandas as pd
import numpy as np
import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, State, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import statsmodels.api as sm
from components.tabs.home import render_home
from components.tabs.tab2 import render_tab2
from components.tabs.tab1 import render_tab1
from components.tabs.tab3 import render_tab3
from components.tabs.tab4 import generate_backtesting_charts
from components.tabs.tab5 import generate_best_model_table
#import components.data as data
from components.data import load_and_process_uploaded_data
import base64
from io import StringIO
#from components.functions import train_default_models, train_optimized_models
from components.functions import get_or_generate_results
import joblib
import os
#from dash import callback_context
from dash import html


# ------------------- Initialize Saved Results -------------------------

try:
    results = joblib.load('results.pkl') 
    print("Results successfully loaded.")
    print("Best Models:", results.get("best_models"))
except FileNotFoundError:
    print("Error: results.pkl not found. Generate the results before running the dashboard.")
    results = None


# -------------------- Initialize the app --------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Data Insight Service"

# Define the layout of the app
app.layout = html.Div([
    html.Header([
        html.H1("BaNex Consulting Data Insight Service", style={'textAlign': 'center', 'fontSize': '40px', 'marginTop': '10px', 'color': 'white'}),
    ], style={'backgroundColor': '#003366', 'padding': '50px'}),

    # Navigation tabs
    dcc.Tabs(
        id='tabs-example', 
        value='home', 
        children=[
            dcc.Tab(label='Business Objectives', value='home'),
            dcc.Tab(label='Data Pre-processing', value='tab-1'),
            dcc.Tab(label='Exploratory Data Analysis', value='tab-2'),
            dcc.Tab(label='Model Performance', value='tab-3'),
            dcc.Tab(label='Model Robustness', value='tab-4'),
            dcc.Tab(label='Business Solution', value='tab-5'),
        ],
        style={
            'fontWeight': 'bold',  # Make the tab names bold
            'fontSize': '16px',  # Increase font size for the tab names
        },
        parent_style={
            'margin': '0 auto',  # Center the tabs
        }
    ),

    # Content section that changes with each tab
    html.Div(id='tabs-content', style={'textAlign': 'center', 'padding': '0px', 'height': '50vh'})
])

# ----------------- Global Variable for Processed Data ------------------

combined_df = None 
arima_df = None

# ----------------- Callback for page content --------------------------

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
        # Pass `results` to render_tab2
        if results is None:
            return html.Div("Error: No results found. Please generate the results and try again.", style={"color": "red"})
        return render_tab2(results)
    
    elif tab == 'tab-3':
        return render_tab3()
    elif tab == 'tab-4':
        if results is None:
            return html.Div("Error: No results found. Please generate the results and try again.", style={"color": "red"})
        
        # Pass results to generate backtesting charts
        return html.Div(generate_backtesting_charts(results))
    
    elif tab == 'tab-5':
        if results is None:
            return html.Div("Error: No results found. Please generate the results and try again.", style={"color": "red"})
        
        # Pass results variable to the table generator
        return generate_best_model_table(results)
    else:
        return html.Div("Invalid tab selected.")

# --------------------- Callback for handling file upload and processing ---------------------

@app.callback(
    [Output('upload-status', 'children'), Output('processed-data-table', 'children')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def process_uploaded_data(contents, filenames):
    global combined_df  # Access the global combined_df variable

    if contents is None:
        return "No files uploaded.", html.Div()

    uploaded_data = {}
    try:
        # Decode and process each uploaded file
        for content, filename in zip(contents, filenames):
            _, content_string = content.split(',')
            decoded = base64.b64decode(content_string).decode('utf-8')
            df = pd.read_csv(StringIO(decoded))
            country_name = filename.split('.')[0]
            uploaded_data[country_name] = df
        
        # Process data and store in the global combined_df variable
        combined_df = load_and_process_uploaded_data(contents, filenames, uploaded_data)
        status_message = "Data processing complete. Displaying processed data below."

        if isinstance(combined_df, pd.DataFrame):
            table = dash_table.DataTable(
                data=combined_df.head(10).to_dict('records'),
                columns=[{"name": i, "id": i} for i in combined_df.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '5px'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )
            return status_message, table
        else:
            return "Error: Processed data is not in the expected format.", html.Div()

    except Exception as e:
        return f"An error occurred during processing: {e}", html.Div()
    



# ------------------ Generate results ----------------------------------------------------------------


@app.callback(
    Output('results-generation-status', 'children'),
    Input('generate-button', 'n_clicks'),
    prevent_initial_call=True
)

def generate_and_save_results(n_clicks):
    #print("Generate button clicked")  # For debugging

    global combined_df 

    # Check if `combined_df` has been processed and exists
    if 'combined_df' not in globals() or combined_df is None:
        return "No processed data available. Please upload your data first."

    results = get_or_generate_results(combined_df)
    return "Results have been generated successfully." if results else "Failed to generate results."


# -------------- Callback for EDA -------------------------
def generate_feature_plot(combined_df, country, feature, allowed_features=None):
    """
    Generate a Plotly figure for a specific country and feature.

    Args:
        data (pd.DataFrame): The dataset containing time-series data with a 'Country' column, 'Time' column, and feature columns.
        country (str): The name of the country to plot.
        feature (str): The feature to visualize.
        allowed_features (list): Optional list of features allowed for plotting.

    Returns:
        plotly.graph_objects.Figure: The generated Plotly figure.
    """
    if combined_df is None:
        raise ValueError("Error: combined_df is None. Ensure the dataset is loaded properly.")

    
    # Specify allowed features if not provided
    allowed_features = allowed_features or [
        'NET Premiums Written',
        'NET Premiums Earned',
        'NET Claims Incurred'
        #'Expenses Incurred',
        #'Total Technical Expenses'
    ]

    # Define a color mapping for features
    feature_colors = {
        'NET Premiums Written': 'blue',
        'NET Premiums Earned': 'green',
        'NET Claims Incurred': 'orange',
        #'Expenses Incurred': 'purple',
        #'Total Technical Expenses': 'orange'
    }

    # Validate the selected feature
    if feature not in allowed_features:
        raise ValueError(f"Feature '{feature}' is not allowed for plotting. Allowed features are: {allowed_features}")

    # Validate if data contains the required columns
    required_columns = ['Country', 'Date', feature]
    for column in required_columns:
        if column not in combined_df.columns:
            raise ValueError(f"Required column '{column}' is missing from the dataset.")

    # Filter the dataset for the selected country
    country_data = combined_df[combined_df['Country'] == country]
    
    if country_data.empty:
        raise ValueError(f"No data found for country: {country}")
    
    # Determine the color for the selected feature
    feature_color = feature_colors.get(feature, 'blue')

    # Generate the plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=country_data['Date'],
        y=country_data[feature],
        mode='lines+markers',
        name=feature,
        line=dict(color=feature_color)
    ))

    # Update layout for better visualization
    fig.update_layout(
        title=f"{feature} Over Time for {country}",
        xaxis=dict(
            title="Date",
            showline=True,   # Ensure the x-axis line is visible
            showgrid=False,  # Remove vertical gridlines
            zeroline=False,   # Remove the zero line
            linecolor='black',   # Set x-axis line color to black
            linewidth=1
        ),
        yaxis=dict(
            title=feature,
            showline=True,   # Ensure the y-axis line is visible
            showgrid=False,  # Remove horizontal gridlines
            zeroline=False,   # Remove the zero line
            linecolor='black',   # Set x-axis line color to black
            linewidth=1
        ),
        template="plotly_white",
        height=500,
        width=800
    )

    return fig

@app.callback(
    Output('feature-plot', 'figure'),  # Note: Return a figure to update the graph
    [Input('country-dropdown', 'value'),
     Input('feature-dropdown', 'value')]
)
def update_plot(selected_country, selected_feature):
    """
    Update the plot based on the selected country and feature.

    Args:
        selected_country (str): The selected country from the dropdown.
        selected_feature (str): The selected feature from the dropdown.

    Returns:
        plotly.graph_objects.Figure: The updated figure.
    """
    global combined_df

    # Define the allowed features
    allowed_features = [
        'NET Premiums Written',
        'NET Premiums Earned',
        'NET Claims Incurred'
        #'Expenses Incurred',
        #'Total Technical Expenses'
    ]

    # Check if both inputs are valid
    if not selected_country or not selected_feature:
        selected_country = combined_df['Country'].iloc[0]
        selected_feature = allowed_features[0]
        # return {
        #     'data': [],
        #     'layout': {
        #         'title': "Please select both a country and a feature.",
        #         'xaxis': {'visible': False},
        #         'yaxis': {'visible': False}
        #     }
        # }

    try:
        # Validate combined_df
        if combined_df is None:
            raise ValueError("Error: combined_df is None. Ensure it is loaded before generating plots.")

        # Validate that the selected feature is allowed
        if selected_feature not in allowed_features:
            raise ValueError(f"Feature '{selected_feature}' is not allowed for plotting.")

        # Ensure combined_df is loaded or passed dynamically
        #combined_df = load_and_process_combined_df()  # Dynamically load or preprocess your DataFrame

        # Generate the plot using the function
        fig = generate_feature_plot(combined_df, selected_country, selected_feature, allowed_features)
        return fig

    except ValueError as e:
        # Return an empty figure with an error message
        return {
            'data': [],
            'layout': {
                'title': f"Error: {str(e)}",
                'xaxis': {'visible': False},
                'yaxis': {'visible': False}
            }
        }



# -------------- Callback for Model Performance -------------------------

@app.callback(
    [Output('model-comparison-graph', 'figure'),
     Output('metrics-bar-chart', 'figure')],
    [Input('tab3-country-dropdown', 'value'),
     Input('tab3-model-dropdown', 'value')]
)
def update_model_predictions(selected_country, selected_models):
    global results
    #print("ARIMA RESULTS", results['country_metrics']["Sweden"])
    # Check if results are loaded
    if results is None:
        raise ValueError("Results have not been generated or loaded. Please ensure results.pkl exists or generate the results.")

    # Handle case where no model is selected
    if not selected_models:
        return go.Figure(), go.Figure()  # Return empty plots if no models are selected

    # Ensure selected_models is a list for multi-selection
    if not isinstance(selected_models, list):
        selected_models = [selected_models]

    # Create figures
    prediction_fig = go.Figure()
    metrics_fig = go.Figure()

    try:
        #print(f"Selected Country: {selected_country}")
        #print(f"Selected Models: {selected_models}")

        # Add the actual values (only once)
        country_metrics = results['country_metrics'].get(selected_country, {})
        first_model = selected_models[0]
        #if first_model:
        # Parse the dataset type and get actual values from the first model
        
        dataset = 'validation'
        start_date = "2022-01-01"
        end_date = "2023-03-31"
        actual_values = list(country_metrics.get("Default XGBoost", {}).get(f'validation_actuals', None))
        date_range = pd.date_range(start=start_date, end=end_date, freq='QS')
        if actual_values is not None and len(actual_values) > 0:
            prediction_fig.add_trace(go.Scatter(
                x=date_range,
                y=list(actual_values.values) if hasattr(actual_values, 'values') else actual_values,
                mode='lines',
                name='Actual',
                line=dict(color='black', dash='dash')  # Black dashed line for actuals
            ))
        
        dataset = 'blind_test'
        start_date = "2023-04-01"
        end_date = "2024-03-31"
        date_range = pd.date_range(start=start_date, end=end_date, freq='QS')
        actual_values = list(country_metrics.get("Default XGBoost", {}).get(f'blind_test_actuals', None))
        if actual_values is not None and len(actual_values) > 0:
            prediction_fig.add_trace(go.Scatter(
                x=date_range,
                y=list(actual_values.values) if hasattr(actual_values, 'values') else actual_values,
                mode='lines',
                #name='Actual',
                showlegend=False,
                line=dict(color='black', dash='dash')  # Black dashed line for actuals
            ))

        

        # Add predictions for each selected model
        for selected_model in selected_models:
            # Parse selected_model to extract the model and dataset
            if 'Validation' in selected_model:
                dataset = 'validation'
                model_name = selected_model.replace(' Validation', '')
                start_date = "2022-01-01"
                end_date = "2023-03-31"
            elif 'Blind Test' in selected_model:
                dataset = 'blind_test'
                model_name = selected_model.replace(' Blind Test', '')
                start_date = "2023-04-01"
                end_date = "2024-03-31"
            else:
                dataset = 'blind_test'
                model_name = selected_model
                start_date = "2023-04-01"
                end_date = "2024-03-31"

            print(f"Parsed Model Name: {model_name}, Dataset: {dataset}")
            date_range = pd.date_range(start=start_date, end=end_date, freq='QS')
            model_metrics = country_metrics.get(model_name, {})
            predictions = model_metrics.get(f'{dataset}_predictions', None)
            if not isinstance(predictions, list):
                predictions = predictions.tolist()
            #print("PREDS", predictions)
            if predictions is not None and len(predictions) > 0:
                #print("DRAWING PREDS for", selected_model)
                prediction_fig.add_trace(go.Scatter(
                    x=date_range,
                    y=predictions,
                    mode='lines',
                    name=f'{selected_model}'
                ))

            # Extract metrics (accuracy, bias)
            metrics_data = country_metrics.get(model_name, {}).get('metrics', {})

            # Determine the dataset type (validation or blind_test) based on the model
            if 'Validation' in selected_model:
                dataset_type = 'validation'
            else:
                dataset_type = 'blind_test'

            # Safely retrieve metrics for the selected dataset
            dataset_metrics = metrics_data.get(dataset_type, {})
            if dataset_metrics:
                # Extract individual metrics
                accuracy = dataset_metrics.get('Accuracy%', 0)
                bias = dataset_metrics.get('Bias%', 0)

                # Add a bar chart trace with these metrics
                metrics_fig.add_trace(go.Bar(
                    x=['Accuracy', 'Bias'],  # Metric names
                    y=[accuracy, bias],     # Metric values
                    name=f"{selected_model} Metrics"
                ))

        # Update the layout for the prediction figure
        prediction_fig.update_layout(
            title=f"Predictions vs Actuals for {selected_country}",
            xaxis_title="Date",
            yaxis_title="NET Claims Incurred",
            legend_title="Legend",
            xaxis=dict(showgrid=False, showline=True, linecolor='black', linewidth=1),
            yaxis=dict(showgrid=False, showline=True, linecolor='black', linewidth=1),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        # Update the layout for the metrics bar chart
        metrics_fig.update_layout(
            title=f"Performance Metrics for {selected_country}",
            yaxis_title="Value (%)",
            barmode='group',  # Group bars by model
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

    except KeyError as e:
        raise ValueError(f"Error accessing data for {selected_country}, {selected_models}: {e}")
    except Exception as e:
        print(e)
    return prediction_fig, metrics_fig







# -------------- Callback for Business Solution -------------------------


@app.callback(
    Output("download-table", "data"),
    [Input("download-button", "n_clicks"),
     #Input("country-dropdown", "value")
     ],  
    prevent_initial_call=True,
)
def download_table(n_clicks):
    global results

    # # Determine which input triggered the callback
    # ctx = callback_context
    # if not ctx.triggered or ctx.triggered[0]["prop_id"].split(".")[0] != "download-button":
    #     return None  # Do nothing if the button was not clicked
    
    quarters = results["prediction_periods"]
     
    if results is None:
        return None
    
    # Extract best models as a DataFrame
    best_models = results.get("best_models", {})
    # Combine the data
    data = [
        {
            "Country": country if i == 0 else "",
            "Best Fit Model for Life LOB": model_info["model"] if i == 0 else "",
            "Year / Quarter": quarters[i],
            "Prediction Values": model_info.get('predictions', [])[i]
            #"Prediction Values": "<br>".join([f"{q}: {p}" for q, p in zip(quarters, map(str, model_info.get('predictions', [])))])
              #join()
        }
        for country, model_info in best_models.items()
        for i in range(len(quarters))
    ]
    df = pd.DataFrame(data)

    # # Filter the DataFrame by the selected countries (multi-select support)
    # if selected_countries:  # Check if any countries are selected
    #     df = df[df["Country"].isin(selected_countries)]

    # # Filter the DataFrame by the selected country if a country is selected
    # if selected_country:
    #     df = df[df["Country"] == selected_country]

    # Convert the DataFrame to a downloadable CSV
    return dcc.send_data_frame(df.to_csv, "Best Fit Models With Prediction Values for Life LOB.csv", index=False)















# -------------- Callback for Back Testing -------------------------

@app.callback(
    [
        Output("accuracy-chart", "figure"),
        Output("bias-chart", "figure"),
        Output("new-plot", "figure"),
        Output("summary-table", "data")  
    ],
    [
        Input("country-dropdown", "value"),
        Input("model-dropdown", "value")
    ]
)
def update_backtesting_charts(selected_country, selected_model):
    # Check for invalid inputs
    try:
        if results is None or selected_country is None or selected_model is None:
            no_data_msg = html.Div("No data available. Please select valid options.", style={"color": "red"})
            return {}, {}, [] , {}
    

        # Retrieve backtesting results
        backtesting_results = results.get("backtesting_results", {})
        country_data = backtesting_results.get(selected_country, {})
        model_data = country_data.get("metrics", {}).get(selected_model, {})
        

        # Filter the summary data for the table
        best_models = results.get("best_models", {})
        summary_data = [
            {
                "Country": country,
                "Best Fit Model for Life LOB": model_info["model"]
            }
            for country, model_info in best_models.items()
            if country == selected_country  
        ]

    
        if not model_data:
            return {}, {}, summary_data , {}
    

        # Process Accuracy Data
        accuracy_based_dict = {}
        for cycle, details in model_data.items():
            for metric in details["metrics"]:
                lag = metric["lag"]
                accuracy = metric["accuracy"]
                lag_key = f"lag {lag}"
                if lag_key not in accuracy_based_dict:
                    accuracy_based_dict[lag_key] = {}
                accuracy_based_dict[lag_key][cycle] = accuracy

        # Process Bias Data
        bias_based_dict = {}
        for cycle, details in model_data.items():
            for metric in details["metrics"]:
                lag = metric["lag"]
                bias = metric["bias"]
                lag_key = f"lag {lag}"
                if lag_key not in bias_based_dict:
                    bias_based_dict[lag_key] = {}
                bias_based_dict[lag_key][cycle] = bias

        # Create Accuracy Chart
        fig_accuracy = go.Figure()
        for cycle in accuracy_based_dict.get("lag 1", {}).keys():
            y_values = [accuracy_based_dict[lag][cycle] for lag in accuracy_based_dict.keys()]
            fig_accuracy.add_trace(go.Bar(
                x=list(accuracy_based_dict.keys()),
                y=y_values,
                name=f"{cycle.capitalize()}",
                marker=dict(line=dict(width=1)),
                width=0.2
            ))

        fig_accuracy.update_layout(
            title=f"Accuracy per Lag per Cycle for {selected_model} in {selected_country}",
            barmode="group",
            bargap=0.4,
            bargroupgap=0.1,
            yaxis_title="Accuracy (%)",
            height=500,
            width=1000,
            plot_bgcolor="white",
            showlegend=True,
        )

        # Create Bias Chart
        fig_bias = go.Figure()
        for cycle in bias_based_dict.get("lag 1", {}).keys():
            y_values = [bias_based_dict[lag][cycle] for lag in bias_based_dict.keys()]
            fig_bias.add_trace(go.Bar(
                x=list(bias_based_dict.keys()),
                y=y_values,
                name=f"{cycle.capitalize()}",
                marker=dict(line=dict(width=1)),
                width=0.2
            ))

        fig_bias.update_layout(
            title=f"Bias per Lag per Cycle for {selected_model} in {selected_country}",
            barmode="group",
            bargap=0.4,
            bargroupgap=0.1,
            yaxis_title="Bias (%)",
            height=500,
            width=1000,
            plot_bgcolor="white",
            showlegend=True,
        )

    # # Create Predictions plot
    # predictions = backtesting_results[country]["predictions"].get(model, {})

    # # Loop through models and collect predictions for each cycle
    # predicted_values = {}  # Store predictions organized by model and cycle
    # for model, cycle_data in all_preds.items():
    #     predicted_values[model] = {cycle: cycle_data[cycle] for cycle in cycles if cycle in cycle_data}
    #backtesting_results[country] = {'predictions': all_preds, 'metrics': all_cycle_metrics, 'preds_actuals': preds_actuals, 'actual_periods': actual_periods}
    
        predictions = backtesting_results[selected_country]["predictions"][selected_model]
        print("SEE", predictions)
        actual_values = backtesting_results[selected_country]["preds_actuals"]
        date_range = backtesting_results[selected_country]["actual_periods"]
        fig_predictions_actuals = go.Figure()
        if actual_values is not None and len(actual_values) > 0:
                fig_predictions_actuals.add_trace(go.Scatter(
                    x=date_range,
                    y=actual_values,
                    mode='lines',
                    name='Actuals',
                    showlegend=True,
                    line=dict(color='black', dash='dash')  # Black dashed line for actuals
                ))

        i = 0
        for cycle, preds in predictions.items():
            x = len(date_range)-i
            periods = date_range[x-4:x]

            # Plot actual values
            fig_predictions_actuals.add_trace(go.Scatter(
                x=periods,
                y=preds,
                mode="lines",
                name=f"{cycle}",
                #line=dict(dash="dash"),
            ))
            i +=1
    

        fig_predictions_actuals.update_layout(
            title="Model Back Testing Predictions vs Actuals Across Cycles",
            xaxis_title="Date",
            yaxis_title="Values",
            plot_bgcolor="white",
            height=500,
            width=1000,
            showlegend=True
        )
    except Exception as e:
        print("THIRD", e)

    return fig_accuracy, fig_bias, fig_predictions_actuals, summary_data


# @app.callback(
#     [
#         Output("accuracy-chart", "figure"),
#         Output("bias-chart", "figure"),
#         #Output("new-plot", "figure"),
#         Output("summary-table", "data")  # Update the table data dynamically
#     ],
#     [
#         Input("country-dropdown", "value"),
#         Input("model-dropdown", "value")
#     ]
# )
# def update_backtesting_charts(selected_country, selected_model):
#     # Check for invalid inputs
#     if results is None or selected_country is None or selected_model is None:
#         no_data_msg = html.Div("No data available. Please select valid options.", style={"color": "red"})
#         return {}, {}, []

#     # Retrieve backtesting results
#     backtesting_results = results.get("backtesting_results", {})
#     country_data = backtesting_results.get(selected_country, {})
#     model_data = country_data.get("metrics", {}).get(selected_model, {})

#     # Filter the summary data for the table
#     best_models = results.get("best_models", {})
#     summary_data = [
#         {
#             "Country": country,
#             "Best Fit Model for Life LOB": model_info["model"]
#         }
#         for country, model_info in best_models.items()
#         if country == selected_country  # Filter by selected country
#     ]

#     if not model_data:
#         return {}, {}, summary_data  # Return filtered table data even if charts are empty

#     # Process Accuracy Data
#     accuracy_based_dict = {}
#     for cycle, details in model_data.items():
#         for metric in details["metrics"]:
#             lag = metric["lag"]
#             accuracy = metric["accuracy"]
#             lag_key = f"lag {lag}"
#             if lag_key not in accuracy_based_dict:
#                 accuracy_based_dict[lag_key] = {}
#             accuracy_based_dict[lag_key][cycle] = accuracy

#     # Process Bias Data
#     bias_based_dict = {}
#     for cycle, details in model_data.items():
#         for metric in details["metrics"]:
#             lag = metric["lag"]
#             bias = metric["bias"]
#             lag_key = f"lag {lag}"
#             if lag_key not in bias_based_dict:
#                 bias_based_dict[lag_key] = {}
#             bias_based_dict[lag_key][cycle] = bias

#     # Create Accuracy Chart
#     fig_accuracy = go.Figure()
#     for cycle in accuracy_based_dict.get("lag 1", {}).keys():
#         y_values = [accuracy_based_dict[lag][cycle] for lag in accuracy_based_dict.keys()]
#         fig_accuracy.add_trace(go.Bar(
#             x=list(accuracy_based_dict.keys()),
#             y=y_values,
#             name=f"{cycle.capitalize()}",
#             marker=dict(line=dict(width=1)),
#             width=0.2
#         ))

#     fig_accuracy.update_layout(
#         title=f"Accuracy per Lag per Cycle for {selected_model} in {selected_country}",
#         barmode="group",
#         bargap=0.4,
#         bargroupgap=0.1,
#         yaxis_title="Accuracy (%)",
#         height=500,
#         width=1000,
#         plot_bgcolor="white",
#         showlegend=True,
#     )

#     # Create Bias Chart
#     fig_bias = go.Figure()
#     for cycle in bias_based_dict.get("lag 1", {}).keys():
#         y_values = [bias_based_dict[lag][cycle] for lag in bias_based_dict.keys()]
#         fig_bias.add_trace(go.Bar(
#             x=list(bias_based_dict.keys()),
#             y=y_values,
#             name=f"{cycle.capitalize()}",
#             marker=dict(line=dict(width=1)),
#             width=0.2
#         ))

#     fig_bias.update_layout(
#         title=f"Bias per Lag per Cycle for {selected_model} in {selected_country}",
#         barmode="group",
#         bargap=0.4,
#         bargroupgap=0.1,
#         yaxis_title="Bias (%)",
#         height=500,
#         width=1000,
#         plot_bgcolor="white",
#         showlegend=True,
#     )

#     return fig_accuracy, fig_bias, summary_data











# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
