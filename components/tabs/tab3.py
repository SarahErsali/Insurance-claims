from dash import dcc, html
import plotly.graph_objs as go

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

        # Instructions for model selection dropdown
        html.P("Select the models and datasets (Validation or Blind Test) you would like to evaluate", style={
            'textAlign': 'left', 
            'fontSize': '20px', 
            'marginTop': '4px', 
            'marginBottom': '5px'
        }),

        # Updated dropdown with options for each combination of default and optimized models on validation and blind test
        dcc.Dropdown(
            id='model-dropdown-prediction',
            options=[
                {'label': 'Default XGBoost (Validation)', 'value': 'xgb_default_val'},
                {'label': 'Default XGBoost (Blind Test)', 'value': 'xgb_default_blind_test'},
                {'label': 'Optimized XGBoost', 'value': 'xgb_optimized'},
                {'label': 'Default LightGBM (Validation)', 'value': 'lgb_default_val'},
                {'label': 'Default LightGBM (Blind Test)', 'value': 'lgb_default_blind_test'},
                {'label': 'Optimized LightGBM', 'value': 'lgb_optimized'},
            ],
            value=['xgb_default_val', 'lgb_default_val'],  # Set default selections
            multi=True,
            style={
                'width': '500px',
                'display': 'inline-block',
                'marginTop': '20px',
                'marginLeft': '-10cm',
                'marginBottom': '15px'
            }
        ),

        # Graph placeholder for model comparisons
        dcc.Graph(id='model-comparison-graph'),
    ])
