from dash import dcc, html



def generate_backtesting_charts(results):
    """
    Generate a layout with dropdowns for country and model selection and placeholders for backtesting charts.

    Args:
        results (dict): The results dictionary containing backtesting metrics.

    Returns:
        html.Div: A Dash layout containing dropdowns and a unified container for charts.
    """
    backtesting_results = results.get("backtesting_results", {})

    # Extract available countries and models
    countries = list(backtesting_results.keys())
    models = list(next(iter(backtesting_results.values())).get("metrics", {}).keys()) if countries else []

    # Layout
    return html.Div(
        [
            # Left-side container for dropdowns
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(
                                "Select Country:",
                                style={
                                    "fontSize": "18px",
                                    "fontWeight": "bold",
                                    "lineHeight": "2",
                                    "textAlign": "left",
                                    "marginBottom": "10px",
                                }
                            ),
                            dcc.Dropdown(
                                id="country-dropdown",
                                options=[{"label": country, "value": country} for country in countries],
                                value=countries[0] if countries else None,
                                placeholder="Select a country",
                                style={
                                    "width": "150%",
                                    "fontSize": "14px",
                                    "padding": "8px",
                                    "height": "50px",
                                }
                            ),
                        ],
                        style={"marginBottom": "50px"}
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Select Model:",
                                style={
                                    "fontSize": "18px",
                                    "fontWeight": "bold",
                                    "lineHeight": "2",
                                    "textAlign": "left",
                                    "marginBottom": "10px",
                                }
                            ),
                            dcc.Dropdown(
                                id="model-dropdown",
                                options=[{"label": model, "value": model} for model in models],
                                value=models[0] if models else None,
                                placeholder="Select a model",
                                style={
                                    "width": "160%",
                                    "fontSize": "14px",
                                    "padding": "8px",
                                    "height": "50px",
                                }
                            ),
                        ],
                        style={"marginBottom": "30px"}
                    ),
                ],
                style={
                    "width": "25%",
                    "padding": "20px",
                    "boxShadow": "0 0 10px rgba(0,0,0,0.1)",  # Add shadow
                    "borderRadius": "10px",  # Add rounded corners
                    "backgroundColor": "white",  # Ensure white background
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "flex-start",  # Align everything to the left
                }
            ),

            # right-side container for the plots
            html.Div(
                [
                    # Placeholder for the cycle chart
                    dcc.Graph(
                        id="cycle-chart-container",
                        config={'displayModeBar': True},
                        style={
                            "width": "48%",  # Adjust width for side-by-side display
                            "display": "inline-block",
                        }
                    ),
                    # Placeholder for the lag chart
                    dcc.Graph(
                        id="lag-chart-container",
                        config={'displayModeBar': True},
                        style={
                            "width": "48%",  # Adjust width for side-by-side display
                            "display": "inline-block",
                            "marginLeft": "4%",  # Add spacing between the two charts
                        }
                    ),
                ],
                style={
                    "width": "75%",  # Adjust width of the plot container
                    "padding": "20px",
                    "boxShadow": "0 0 10px rgba(0,0,0,0.1)",  # Add shadow around the entire right container
                    "borderRadius": "10px",  # Rounded corners for the container
                    "backgroundColor": "white",  # Ensure white background
                    "display": "flex",
                    "flexDirection": "row",  # Align charts side by side
                    "justifyContent": "space-between",  # Add spacing between charts
                    "alignItems": "center",  # Vertically center align the charts
                }
            ),
        ],
        style={
            "display": "flex",
            "flexDirection": "row",
            "height": "100vh",
            "padding": "20px",
            "gap": "20px",  # Add spacing between the left and right containers
            "backgroundColor": "#f9f9f9",  # Light gray background for overall layout
        }
    )








# def generate_backtesting_charts(results):
#     """
#     Generate a layout with dropdowns for country and model selection and a placeholder for backtesting charts.

#     Args:
#         results (dict): The results dictionary containing backtesting metrics.

#     Returns:
#         html.Div: A Dash layout containing dropdowns and an empty chart container.
#     """
#     backtesting_results = results.get("backtesting_results", {})

#     # Extract available countries and models
#     countries = list(backtesting_results.keys())
#     models = list(next(iter(backtesting_results.values())).keys()) if countries else []

#     # Layout
#     return html.Div(
#         [
#             # Left-side container for dropdowns
#             html.Div(
#                 [
#                     html.Div(
#                         [
#                             html.Label(
#                                 "Select Country:",
#                                 style={
#                                     "fontSize": "18px",
#                                     "fontWeight": "bold",
#                                     "lineHeight": "2",  # Adjusted for consistency
#                                     "textAlign": "left",  # Left-aligned label
#                                     "marginBottom": "10px",  # Spacing below label
#                                 }
#                             ),
#                             dcc.Dropdown(
#                                 id="country-dropdown",
#                                 options=[{"label": country, "value": country} for country in countries],
#                                 value=countries[0] if countries else None,
#                                 placeholder="Select a country",
#                                 style={
#                                     "width": "100%",
#                                     "fontSize": "16px",  # Larger font size
#                                     "padding": "10px",  # Increased padding
#                                     "height": "50px",  # Adjust height
#                                 }
#                             ),
#                         ],
#                         style={"marginBottom": "50px"}  # Increased space between dropdowns
#                     ),
#                     html.Div(
#                         [
#                             html.Label(
#                                 "Select Model:",
#                                 style={
#                                     "fontSize": "18px",
#                                     "fontWeight": "bold",
#                                     "lineHeight": "2",  # Adjusted for consistency
#                                     "textAlign": "left",  # Left-aligned label
#                                     "marginBottom": "10px",  # Spacing below label
#                                 }
#                             ),
#                             dcc.Dropdown(
#                                 id="model-dropdown",
#                                 options=[{"label": model, "value": model} for model in models],
#                                 value=models[0] if models else None,
#                                 placeholder="Select a model",
#                                 style={
#                                     "width": "100%",
#                                     "fontSize": "16px",  # Larger font size
#                                     "padding": "10px",  # Increased padding
#                                     "height": "50px",  # Adjust height
#                                 }
#                             ),
#                         ],
#                         style={"marginBottom": "30px"}  # Increased space between dropdowns
#                     ),
#                 ],
#                 style={
#                     "width": "30%",  # Adjust width of dropdown container
#                     "padding": "20px",  # Add some padding
#                     "boxShadow": "0 0 10px rgba(0,0,0,0.1)",  # Add light shadow
#                     "textAlign": "justify",  # Justify text alignment
#                 }
#             ),

#             # Right-side container for the plot
#             html.Div(
#                 id="backtesting-chart-container",
#                 style={
#                     "width": "70%",  # Adjust width of plot container
#                     "display": "flex",
#                     "justifyContent": "center",
#                     "alignItems": "center",
#                     "padding": "20px",
#                 }
#             )
#         ],
#         style={
#             "display": "flex",  # Align dropdowns and plot side by side
#             "flexDirection": "row",
#             "height": "100vh",  # Make the container take the full height of the viewport
#         }
#     )
