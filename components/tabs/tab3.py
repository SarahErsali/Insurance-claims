from dash import dcc, html
import plotly.graph_objs as go
import joblib
import os
from components.functions import get_or_generate_results


# Load results from file
results_file = 'results.pkl'
if os.path.exists(results_file):
    results = joblib.load(results_file)
    print("Results successfully loaded in tab3.")
else:
    print(f"Warning: '{results_file}' not found in tab3. Generate the results using the 'Generate' button.")
    results = None

def render_tab3():
    """
    Render the layout for Tab 3, allowing users to select a country and model to view comparison graphs.
    """
    # Extract the list of countries from results
    countries = list(results.get('country_metrics', {}).keys()) if results else []
    
    # Set the default country to the first one in the list
    default_country = countries[0] if countries else None

    return html.Div(
        [
            # Container for dropdowns (left side)
            html.Div(
                [
                    # Dropdown for country selection
                    html.Div(
                        [
                            html.Label(
                                "Select Country:", 
                                style={'fontWeight': 'bold', 'fontSize': '18px', 'textAlign': 'left'}
                            ),
                            dcc.Dropdown(
                                id='tab3-country-dropdown',
                                options=[{'label': country, 'value': country} for country in countries],
                                value=default_country,  # Set the first country as the default
                                style={
                                    'width': '300px',
                                    'marginBottom': '20px',
                                    'fontSize': '14px',
                                    'padding': '8px',
                                }
                            ),
                        ],
                        style={'textAlign': 'left'}
                    ),

                    # Dropdown for model and dataset selection
                    html.Div(
                        [
                            html.Label(
                                "Select Model:", 
                                style={'fontWeight': 'bold', 'fontSize': '18px', 'textAlign': 'left'}
                            ),
                            dcc.Dropdown(
                                id='tab3-model-dropdown',
                                options=[
                                    {'label': 'Default XGBoost (Validation)', 'value': 'Default XGBoost Validation'},
                                    {'label': 'Default XGBoost (Blind Test)', 'value': 'Default XGBoost Blind Test'},
                                    {'label': 'Default LightGBM (Validation)', 'value': 'Default LightGBM Validation'},
                                    {'label': 'Default LightGBM (Blind Test)', 'value': 'Default LightGBM Blind Test'},
                                    {'label': 'Retrained XGBoost (Blind Test)', 'value': 'Retrained XGBoost Blind Test'},
                                    {'label': 'Retrained LightGBM (Blind Test)', 'value': 'Retrained LightGBM Blind Test'},
                                    {'label': 'ARIMA (Blind Test)', 'value': 'ARIMA'},
                                    {'label': 'Moving Average (Blind Test)', 'value': 'Moving Average'}
                                ],
                                value='Default XGBoost Validation',
                                style={
                                    'width': '300px',
                                    'fontSize': '14px',
                                    'padding': '8px',
                                }
                            ),
                        ],
                        style={'textAlign': 'left'}
                    ),
                ],
                style={
                    'width': '25%',
                    'padding': '20px',
                    'boxShadow': '0 0 10px rgba(0,0,0,0.1)',
                    'display': 'flex',
                    'flexDirection': 'column',
                    'alignItems': 'flex-start',
                    'backgroundColor': 'white',
                    'borderRadius': '10px',
                    'marginRight': '20px',
                    'height': '95vh',  # Increased height to fit both plots
                }
            ),

            # Graph container (right side)
            html.Div(
                [
                    dcc.Graph(
                        id='model-comparison-graph',
                        config={'displayModeBar': True}
                    ),
                    html.Div(
                        dcc.Graph(id='metrics-bar-chart'),
                        style={'marginTop': '20px'}
                    )
                ],
                style={
                    'width': '75%',
                    'padding': '20px',
                    'backgroundColor': 'white',
                    'borderRadius': '10px',
                    'boxShadow': '0 0 10px rgba(0, 0, 0, 0.1)',
                    'height': '95vh',  # Increased height to fit both plots
                    'overflowY': 'auto',  # Enable vertical scrolling if necessary
                }
            ),
        ],
        style={
            'display': 'flex',
            'flexDirection': 'row',
            'height': '100vh',
            'backgroundColor': 'white',
            'padding': '20px',
        }
    )

# def render_tab3():
#     """
#     Render the layout for Tab 3, allowing users to select a country and model to view comparison graphs.
#     """
#     # Extract the list of countries from results
#     countries = list(results.get('country_metrics', {}).keys()) if results else []
    
#     # Set the default country to the first one in the list
#     default_country = countries[0] if countries else None

#     return html.Div(
#         [
#             # Container for dropdowns (left side)
#             html.Div(
#                 [
#                     # Dropdown for country selection
#                     html.Div(
#                         [
#                             html.Label(
#                                 "Select Country:", 
#                                 style={'fontWeight': 'bold', 'fontSize': '18px', 'textAlign': 'left'}
#                             ),
#                             dcc.Dropdown(
#                                 id='tab3-country-dropdown',
#                                 options=[{'label': country, 'value': country} for country in countries],
#                                 value=default_country,  # Set the first country as the default
#                                 style={
#                                     'width': '300px',
#                                     'marginBottom': '20px',
#                                     'fontSize': '14px',  # Adjust font size for consistency
#                                     'padding': '8px',  # Add padding for dropdown
#                                 }
#                             ),
#                         ],
#                         style={'textAlign': 'left'}  # Ensure left alignment for the container
#                     ),

#                     # Dropdown for model and dataset selection
#                     html.Div(
#                         [
#                             html.Label(
#                                 "Select Model:", 
#                                 style={'fontWeight': 'bold', 'fontSize': '18px', 'textAlign': 'left'}
#                             ),
#                             dcc.Dropdown(
#                                 id='tab3-model-dropdown',
#                                 options=[
#                                     {'label': 'Default XGBoost (Validation)', 'value': 'Default XGBoost Validation'},
#                                     {'label': 'Default XGBoost (Blind Test)', 'value': 'Default XGBoost Blind Test'},
#                                     {'label': 'Default LightGBM (Validation)', 'value': 'Default LightGBM Validation'},
#                                     {'label': 'Default LightGBM (Blind Test)', 'value': 'Default LightGBM Blind Test'},
#                                     {'label': 'Retrained XGBoost (Blind Test)', 'value': 'Retrained XGBoost Blind Test'},
#                                     {'label': 'Retrained LightGBM (Blind Test)', 'value': 'Retrained LightGBM Blind Test'},
#                                     {'label': 'ARIMA (Blind Test)', 'value': 'ARIMA'},
#                                     {'label': 'Moving Average (Blind Test)', 'value': 'Moving Average'}
#                                 ],
#                                 value='Default XGBoost Validation',  # Default selection
#                                 style={
#                                     'width': '300px',
#                                     'fontSize': '14px',  # Adjust font size for consistency
#                                     'padding': '8px',  # Add padding for dropdown
#                                 }
#                             ),
#                         ],
#                         style={'textAlign': 'left'}  # Ensure left alignment for the container
#                     ),
#                 ],
#                 style={
#                     'width': '25%',  # Adjust width of dropdown container
#                     'padding': '20px',
#                     'boxShadow': '0 0 10px rgba(0,0,0,0.1)',  # Add light shadow for dropdown container
#                     'display': 'flex',
#                     'flexDirection': 'column',
#                     'alignItems': 'flex-start',  # Align everything to the left
#                     'backgroundColor': 'white',  # Ensure white background for the dropdown section
#                     'borderRadius': '10px',  # Optional: Rounded corners for dropdown container
#                     'marginRight': '20px',  # Add spacing between dropdown and graph
#                 }
#             ),

#             # Graph container (right side)
#             html.Div(
#                 dcc.Graph(
#                     id='model-comparison-graph',
#                     config={'displayModeBar': True},  # Display Plotly toolbar
#                 ),
#                 style={
#                     'width': '75%',  # Further increased width of the graph
#                     'padding': '20px',
#                     'backgroundColor': 'white',  # White background for the graph
#                     'borderRadius': '10px',  # Optional: Rounded corners for better aesthetics
#                     'boxShadow': '0 0 10px rgba(0, 0, 0, 0.1)',  # Add shadow for the graph container
#                 }
#             ),
#         ],
#         style={
#             'display': 'flex',
#             'flexDirection': 'row',
#             'height': '100vh',
#             'backgroundColor': 'white',  # Set entire page background to white
#             'padding': '20px',  # Add padding around the entire layout
#         }
#     )


