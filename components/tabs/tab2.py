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
        #'Expenses Incurred',
        #'Total Technical Expenses'
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
                                options=[{'label': country, 'value': country}for country in countries],
                                value=default_country,  # Set the default country
                                #placeholder="Select a country",  # Optional placeholder
                                style={
                                    'width': '300px',
                                    'marginBottom': '20px',
                                    'fontSize': '14px',  # Adjust font size for consistency
                                    'padding': '8px',  # Add padding for dropdown
                                }
                            ),
                        ],
                        style={'textAlign': 'left'}  # Ensure left alignment for the container
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
                                value=default_feature,  # Set the default feature
                                #placeholder="Select a feature",  # Optional placeholder
                                style={
                                    'width': '300px',
                                    'fontSize': '14px',  # Adjust font size for consistency
                                    'padding': '8px',  # Add padding for dropdown
                                }
                            ),
                        ],
                        style={'textAlign': 'left'}  # Ensure left alignment for the container
                    ),
                ],
                style={
                    'width': '25%',  # Adjust width of dropdown container
                    'padding': '20px',
                    'boxShadow': '0 0 10px rgba(0,0,0,0.1)',  # Add light shadow for dropdown container
                    'display': 'flex',
                    'flexDirection': 'column',
                    'alignItems': 'flex-start',  # Align everything to the left
                    'backgroundColor': 'white',  # Ensure white background for the dropdown section
                    'borderRadius': '10px',  # Optional: Rounded corners for dropdown container
                    'marginRight': '20px',  # Add spacing between dropdown and graph
                }
            ),

            # Graph container (right side)
            html.Div(
                dcc.Graph(
                    id='feature-plot',
                ),
                style={
                    'width': '75%',  # Adjust width of the graph container
                    'padding': '20px',
                    'backgroundColor': 'white',  # White background for the graph
                    'borderRadius': '10px',  # Optional: Rounded corners for better aesthetics
                    'boxShadow': '0 0 10px rgba(0, 0, 0, 0.1)',  # Add shadow for the graph container
                }
            ),
        ],
        style={
            'display': 'flex',
            'flexDirection': 'row',
            'height': '100vh',
            'backgroundColor': 'white',  # Set entire page background to white
            'padding': '20px',  # Add padding around the entire layout
        }
    )
