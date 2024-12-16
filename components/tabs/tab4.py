from dash import dcc, html, dash_table



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

    #-------
    # Extract summary data for the table
    best_models = results.get("best_models", {})
    summary_data = [
        {"Country": country, "Best Fit Model for Life LOB": model_info["model"]}
        for country, model_info in best_models.items()
    ]

    # Create summary table
    summary_table = dash_table.DataTable(
        id="summary-table",
        columns=[
            {"name": "Country", "id": "Country"},
            {"name": "Best Fit Model for Life LOB", "id": "Best Fit Model for Life LOB"},
        ],
        data=summary_data,  # Pre-populate with summary data
        style_table={"width": "100%", "marginBottom": "20px"},  # Full width and spacing below the table
        style_cell={
            "textAlign": "center",
            "padding": "8px",
            "fontFamily": "Arial",
            "fontSize": "14px",
        },
        style_header={
            "backgroundColor": "rgb(230, 230, 230)",
            "fontWeight": "bold",
        },
    )

    
    #-----

    # Layout
    return html.Div(
        [
            # Left-side container for table and dropdowns
            html.Div(
                [
                    # Add the summary table
                    html.Div(summary_table, style={"marginTop": "30px", "marginBottom": "30px", "marginLeft": "auto", "marginRight": "auto", "textAlign": "center", "width": "90%",}),

                    # Dropdown for selecting country
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
                                    "width": "160%",
                                    "fontSize": "14px",
                                    "padding": "8px",
                                    "height": "50px",
                                    "textAlign": "left",
                                }
                            ),
                        ],
                        style={"marginBottom": "30px"}
                    ),

                    # Dropdown for selecting model
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
                                    "width": "167%",
                                    "fontSize": "14px",
                                    "padding": "14px",
                                    "height": "50px",
                                    "textAlign": "left",
                                }
                            ),
                        ],
                        style={"marginBottom": "30px"}
                    ),
                ],
                style={
                    "width": "27%",
                    "padding": "20px",
                    "boxShadow": "0 0 10px rgba(0,0,0,0.1)",  # Add shadow
                    "borderRadius": "10px",  # Add rounded corners
                    "backgroundColor": "white",  # Ensure white background
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "flex-start",  # Align everything to the left
                    "height": "95vh",  # Increased height to fit both plots
                }
            ),

            # Right-side container for the plots
            html.Div(
                [
                    # Container for the accuracy chart
                    html.Div(
                        dcc.Graph(
                            id="accuracy-chart",
                            config={"displayModeBar": True},
                            style={
                                "width": "100%",  
                                "height": "40vh",  # Set a fixed height
                            }
                        ),
                        style={"marginBottom": "250px"}  # Explicit spacing for this chart
                    ),
                    # Container for the bias chart
                    html.Div(
                        dcc.Graph(
                            id="bias-chart",
                            config={"displayModeBar": True},
                            style={
                                "width": "100%",  
                                "height": "40vh",  # Set a fixed height
                            }
                        )
                    ),
                ],
                style={
                    "width": "75%",  # Adjust width of the plot container
                    "padding": "20px",
                    "boxShadow": "0 0 10px rgba(0,0,0,0.1)",  # Add shadow around the entire right container
                    "borderRadius": "10px",  # Rounded corners for the container
                    "backgroundColor": "white",  # Ensure white background
                    "display": "flex",
                    "flexDirection": "column",  # Stack charts vertically
                    "justifyContent": "flex-start",  # Align charts at the top
                    "alignItems": "center",  # Center align the charts horizontally
                    "height": "95vh",  # Increased height to fit both plots
                    "overflowY": "auto",  # Enable vertical scrolling if necessary
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

   





















