from dash import dcc, html
import plotly.graph_objs as go
import joblib
from components.functions import get_or_generate_results


# Load results from file
results_file = 'results.pkl'
results = joblib.load(results_file)

def render_tab3():
    return html.Div([
        # Title and description for the section
        html.H3("Models Evaluation", style={
            'fontWeight': 'bold',
            'textAlign': 'left',
            'fontSize': '30px',
            'marginTop': '3cm'
        }),

        html.P(
            "The line plot shows the predictions of various models—XGBoost, LightGBM, and their optimized versions—"
            "against the actual values for 'Claims Incurred' over a period. The dotted black line represents the actual observed values, "
            "while the colored lines show the predictions of the models.",
            style={
                'fontSize': '16px',
                'textAlign': 'left',
                'lineHeight': '2.0',
            }
        ),

        # Bullet points explaining the models
        html.Ul([
            html.Li([
                html.Span("XGBoost: ", style={'fontWeight': 'bold'}),
                "In flatter regions or moderate fluctuations, XGBoost tends to smooth out predictions but follows the overall pattern with smaller deviations."
            ], style={'fontSize': '16px', 'textAlign': 'left', 'lineHeight': '2'}),

            html.Li([
                html.Span("LightGBM: ", style={'fontWeight': 'bold'}),
                "LightGBM appears less stable, with larger prediction errors in regions with moderate claim variations."
            ], style={'fontSize': '16px', 'textAlign': 'left', 'lineHeight': '2'}),
        ], style={'textAlign': 'left', 'marginBottom': '2.5cm'}),

        # Instructions for selecting country, model, and dataset
        html.P("Select the country, model, and dataset (Validation or Blind Test) you would like to evaluate:", style={
            'textAlign': 'left',
            'fontSize': '20px',
            'marginTop': '4px',
            'marginBottom': '10px'
        }),

        # Dropdown for country selection
        html.Div([
            html.Label("Country:", style={'fontWeight': 'bold', 'fontSize': '16px'}),
            dcc.Dropdown(
                id='tab3-country-dropdown',
                options=[
                    {'label': 'All Countries', 'value': 'All Countries'}
                ] + [{'label': country, 'value': country} for country in results['country_metrics'].keys()],
                value='All Countries',  # Default selection
                style={'width': '300px', 'display': 'inline-block', 'marginRight': '20px'}
            ),

            # Dropdown for model selection
            html.Label("Model:", style={'fontWeight': 'bold', 'fontSize': '16px'}),
            dcc.Dropdown(
                id='tab3-model-dropdown',
                options=[
                    {'label': 'Default XGBoost', 'value': 'Default XGBoost'},
                    {'label': 'Retrained XGBoost', 'value': 'Retrained XGBoost'},
                    {'label': 'Default LightGBM', 'value': 'Default LightGBM'},
                    {'label': 'Retrained LightGBM', 'value': 'Retrained LightGBM'},
                    {'label': 'ARIMA', 'value': 'ARIMA'},
                    {'label': 'Moving Average', 'value': 'Moving Average'}
                ],
                value='Default XGBoost',  # Default selection
                style={'width': '300px', 'display': 'inline-block', 'marginRight': '20px'}
            ),

            # Dropdown for data selection
            html.Label("Dataset:", style={'fontWeight': 'bold', 'fontSize': '16px'}),
            dcc.Dropdown(
                id='tab3-data-dropdown',
                options=[
                    {'label': 'Validation Data', 'value': 'validation'},
                    {'label': 'Blind Test Data', 'value': 'blind_test'}
                ],
                value='validation',  # Default selection
                style={'width': '300px', 'display': 'inline-block'}
            ),
        ], style={'marginBottom': '20px'}),

        # Graph placeholder for model comparisons
        dcc.Graph(id='model-comparison-graph'),
    ])


# def render_tab3():
#     return html.Div([
#         # Title and description for the section
#         html.H3("Models Evaluation", style={
#             'fontWeight': 'bold', 
#             'textAlign': 'left', 
#             'fontSize': '30px', 
#             'marginTop': '3cm'
#         }),

#         html.P(
#             "The line plot shows the predictions of various models—XGBoost, LightGBM, and their optimized versions—"
#             "against the actual values for 'Claims Incurred' over a period. The dotted black line represents the actual observed values, "
#             "while the colored lines show the predictions of the models.",
#             style={
#                 'fontSize': '16px', 
#                 'textAlign': 'left', 
#                 'lineHeight': '2.0',
#             }
#         ),

#         # Bullet points explaining the models
#         html.Ul([
#             html.Li([
#                 html.Span("XGBoost: ", style={'fontWeight': 'bold'}),
#                 "In flatter regions or moderate fluctuations, XGBoost tends to smooth out predictions but follows the overall pattern with smaller deviations."
#             ], style={'fontSize': '16px', 'textAlign': 'left', 'lineHeight': '2'}),
            
#             html.Li([
#                 html.Span("LightGBM: ", style={'fontWeight': 'bold'}),
#                 "LightGBM appears less stable, with larger prediction errors in regions with moderate claim variations."
#             ], style={'fontSize': '16px', 'textAlign': 'left', 'lineHeight': '2'}),
#         ], style={'textAlign': 'left', 'marginBottom': '2.5cm'}),

#         # Instructions for model selection dropdown
#         html.P("Select the models and datasets (Validation or Blind Test) you would like to evaluate", style={
#             'textAlign': 'left', 
#             'fontSize': '20px', 
#             'marginTop': '4px', 
#             'marginBottom': '5px'
#         }),

#         # Updated dropdown with options for each combination of default and optimized models on validation and blind test
#         dcc.Dropdown(
#             id='model-dropdown-prediction',
#             options=[
#                 {'label': 'Default XGBoost (Validation)', 'value': 'xgb_default_val'},
#                 {'label': 'Default XGBoost (Blind Test)', 'value': 'xgb_default_blind_test'},
#                 {'label': 'Optimized XGBoost', 'value': 'xgb_optimized'},
#                 {'label': 'Default LightGBM (Validation)', 'value': 'lgb_default_val'},
#                 {'label': 'Default LightGBM (Blind Test)', 'value': 'lgb_default_blind_test'},
#                 {'label': 'Optimized LightGBM', 'value': 'lgb_optimized'},
#             ],
#             value=['xgb_default_val', 'lgb_default_val'],  # Set default selections
#             multi=True,
#             style={
#                 'width': '500px',
#                 'display': 'inline-block',
#                 'marginTop': '20px',
#                 'marginLeft': '-10cm',
#                 'marginBottom': '15px'
#             }
#         ),

#         # Graph placeholder for model comparisons
#         dcc.Graph(id='model-comparison-graph'),
#     ])
