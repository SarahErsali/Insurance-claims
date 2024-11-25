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
app.title = "Insurance Consultant Service"

# Define the layout of the app
app.layout = html.Div([
    html.Header([
        html.H1("Wellcome to BaNex Consulting Insurance Service", style={'textAlign': 'center', 'fontSize': '48px', 'marginTop': '10px'}),
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
        return render_tab2()
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




# -------------- Callback for Model Performance -------------------------


@app.callback(
    Output('model-comparison-graph', 'figure'),
    Input('tab3-country-dropdown', 'value'),
    Input('tab3-model-dropdown', 'value'),
)
def update_model_predictions(selected_country, selected_model):
    global results

    # Check if results are loaded
    if results is None:
        raise ValueError("Results have not been generated or loaded. Please ensure results.pkl exists or generate the results.")

    # Create a Plotly figure
    fig = go.Figure()

    try:
        # Debugging: Log selected inputs
        print(f"Selected Country: {selected_country}")
        print(f"Selected Model: {selected_model}")

        # Parse selected_model to extract the model and dataset
        if 'Validation' in selected_model:
            dataset = 'validation'
            model_name = selected_model.replace(' Validation', '')
        elif 'Blind Test' in selected_model:
            dataset = 'blind_test'
            model_name = selected_model.replace(' Blind Test', '')
        else:
            dataset = 'blind_test'
            model_name = selected_model  # For ARIMA and Moving Average

        print(f"Parsed Model Name: {model_name}, Dataset: {dataset}")

        # Handle "All Countries" option
        if selected_country == 'All Countries':
            predictions = results['ml_predictions']['all_countries_predictions'].get(model_name, None)
            # For default ML models, fetch actuals from results['ml_predictions']
            if model_name in ['Default XGBoost', 'Default LightGBM']:
                actual_values = results['ml_predictions'][f'{dataset}_actuals'].get(model_name, None)
            else:
                # For retrained ML models, ARIMA, and Moving Average, fetch actuals differently
                actual_values = results['ml_predictions'][f'{dataset}_actuals'].get('Default XGBoost', None)

        else:
            # Get country-specific data
            country_metrics = results['country_metrics'].get(selected_country, {})
            print(f"Country Metrics for {selected_country}: {list(country_metrics.keys())}")  # Debugging statement

            model_metrics = country_metrics.get(model_name, {})
            print(f"Model Metrics for {model_name}: {list(model_metrics.keys())}")  # Debugging statement

            actual_values = model_metrics.get(f'{dataset}_actuals', None)
            predictions = model_metrics.get(f'{dataset}_predictions', None)

        # Debugging: Check retrieved values
        print(f"Actual Values: {actual_values}")
        print(f"Predictions: {predictions}")

        

        # # Raise an error if both predictions and actual values are missing
        # if predictions is None and actual_values is None:
        #     raise KeyError(f"No data found for {selected_country}, {selected_model}.")

        # # Specific debugging for ARIMA and Moving Average
        # if model_name == 'ARIMA':
        #     print(f"ARIMA Metrics: {results['country_metrics'][selected_country]['ARIMA']}")  # Debugging statement
        # elif model_name == 'Moving Average':
        #     print(f"Moving Average Metrics: {results['country_metrics'][selected_country]['Moving Average']}")  # Debugging statement

        # # Align indices for actual values and predictions
        # if hasattr(actual_values, "index") and hasattr(predictions, "index"):
        #     actual_values = actual_values.reindex(predictions.index).dropna()
        #     predictions = predictions.loc[actual_values.index]
        # elif len(actual_values) != len(predictions):
        #     # Fallback to range index if lengths differ
        #     actual_values = pd.Series(actual_values).reset_index(drop=True)
        #     predictions = pd.Series(predictions).reset_index(drop=True)

        # Align indices for actual values and predictions
        if hasattr(actual_values, "index") and hasattr(predictions, "index"):
            # Align actual_values and predictions by their shared indices
            aligned_indices = actual_values.index.intersection(predictions.index)
            actual_values = actual_values.loc[aligned_indices]
            predictions = predictions.loc[aligned_indices]
        elif len(actual_values) != len(predictions):
            # Warn about mismatched lengths
            print("Warning: Actual values and predictions have mismatched lengths.")
            return html.Div("Error: Mismatched lengths between actual values and predictions.", style={"color": "red"})

        # Add actual values to the plot
        if actual_values is not None and len(actual_values) > 0:
            fig.add_trace(go.Scatter(
                x=actual_values.index if hasattr(actual_values, 'index') else list(range(len(actual_values))),
                y=actual_values.values if hasattr(actual_values, 'values') else actual_values,
                mode='lines',
                name='Actual',
                line=dict(dash='dot')
            ))

        # Add model predictions to the plot
        if predictions is not None and len(predictions) > 0:
            fig.add_trace(go.Scatter(
                x=predictions.index if hasattr(predictions, 'index') else list(range(len(predictions))),
                y=predictions.values if hasattr(predictions, 'values') else predictions,
                mode='lines',
                name=f'{selected_model}'
            ))

    except KeyError as e:
        raise ValueError(f"Error accessing data for {selected_country}, {selected_model}: {e}")

    # Update the layout of the figure
    fig.update_layout(
        title=f"{selected_model} vs Actual for {selected_country}",
        xaxis_title="Date",
        yaxis_title="NET Claims Incurred",
        legend_title="Legend",
        template="plotly_dark"
    )

    return fig




# -------------- Callback for Business Solution -------------------------


@app.callback(
    Output("download-table", "data"),
    Input("download-button", "n_clicks"),
    prevent_initial_call=True,
)
def download_table(n_clicks):
    if results is None:
        return None
    
    # Extract best models as a DataFrame
    best_models = results.get("best_models", {})
    data = [{"Country": country, "Best Fit Model for Life LOB Data": model_info["model"]} for country, model_info in best_models.items()]
    df = pd.DataFrame(data)

    # Convert the DataFrame to a downloadable CSV
    return dcc.send_data_frame(df.to_csv, "best_models.csv", index=False)



# -------------- Callback for Back Testing -------------------------


@app.callback(
    Output("backtesting-chart-container", "children"),
    [Input("country-dropdown", "value"),
     Input("model-dropdown", "value")]
)
def update_backtesting_chart(selected_country, selected_model):
    if results is None or selected_country is None or selected_model is None:
        return html.Div("No data available. Please select valid options.", style={"color": "red"})

    backtesting_results = results.get("backtesting_results", {})
    country_data = backtesting_results.get(selected_country, {})
    model_data = country_data.get(selected_model, {})

    # Ensure valid data is available
    if not model_data:
        return html.Div(f"No data available for {selected_country} and {selected_model}.", style={"color": "red"})

    # Create the figure
    fig = go.Figure()
    for metric in ["bias", "accuracy", "mape"]:
        fig.add_trace(go.Bar(
            x=[f"Cycle {i + 1}" for i in range(len(model_data[metric]))],
            y=model_data[metric],
            name=metric.capitalize(),
            # text=model_data[metric],
            # textposition='auto'
        ))

    # Update layout for the figure
    fig.update_layout(
        title=f"Backtesting Metrics for {selected_model} in {selected_country}",
        barmode='group',
        #xaxis_title="Cycles",
        yaxis_title="Value (%)",
        legend_title="Metrics",
        height=500,
        width=800,
        xaxis=dict(showgrid=False),  # Remove x-axis gridlines
        yaxis=dict(showgrid=False),  # Remove y-axis gridlines
        plot_bgcolor="white",       # Set plot background to white
    )

    return dcc.Graph(figure=fig)






# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
