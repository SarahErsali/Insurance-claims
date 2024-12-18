from dash import dcc, html
import plotly.graph_objects as go
#from components.data import combined_df


# def render_tab2():
#     # Placeholder for Model Training tab content
#     return html.Div("This is the Model Training tab.")




def render_tab2(results):
    """
    Renders the layout for the second tab, including dropdowns for country and feature selection,
    and a container for the plot.

    Returns:
        html.Div: The layout for Tab 2.
    """
    backtesting_results = results.get("backtesting_results", {})
    countries = list(backtesting_results.keys())

    # Predefined list of features for plotting
    features = [
        'NET Premiums Written',
        'NET Premiums Earned',
        'NET Claims Incurred'
    ]

    # Default values
    default_country = countries[0] if countries else None
    default_feature = features[0]

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
                                style={
                                    'fontWeight': 'bold', 
                                    'fontSize': '18px', 
                                    'textAlign': 'left'
                                }
                            ),
                            dcc.Dropdown(
                                id='country-dropdown',
                                options=[{'label': country, 'value': country} for country in countries],
                                value=default_country,
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

                    # Dropdown for feature selection
                    html.Div(
                        [
                            html.Label(
                                "Select Feature:", 
                                style={
                                    'fontWeight': 'bold', 
                                    'fontSize': '18px', 
                                    'textAlign': 'left'
                                }
                            ),
                            dcc.Dropdown(
                                id='feature-dropdown',
                                options=[{'label': feature, 'value': feature} for feature in features],
                                value=default_feature,
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
                }
            ),

            # Graph container (right side)
            html.Div(
                [
                    # Descriptive Text
                    html.P(
                        "The following plot provides a visualization of key financial metrics: Net Premiums Written, Net Premiums Incurred, and Net Claims Incurred. "
                        "These metrics are presented for the quarterly preprocessed data of the uploaded countries. "
                        "Explore the data to identify trends, patterns, and seasonality across different countries and time periods.",
                        style={
                            'fontSize': '18px',
                            'lineHeight': '1.8',
                            "marginTop": "30px",
                            'marginBottom': '40px',
                            'textAlign': 'justify',
                        }
                    ),
                    # Plot
                    dcc.Graph(
                        id='feature-plot',
                    )
                ],
                style={
                    'width': '75%',
                    'padding': '20px',
                    'backgroundColor': 'white',
                    'borderRadius': '10px',
                    'boxShadow': '0 0 10px rgba(0, 0, 0, 0.1)',
                    'overflowY': 'auto',  # Enable vertical scrolling
                    'maxHeight': '95vh',  # Limit the height of the container
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

