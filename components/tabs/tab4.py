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
        data=summary_data,
        style_table={"width": "100%", "marginBottom": "20px"},
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

    # Layout
    return html.Div(
        [
            # Left-side container for table and dropdowns
            html.Div(
                [
                    html.Div(summary_table, style={"marginTop": "30px", "marginBottom": "30px", "width": "90%"}),

                    # Dropdown for selecting country
                    html.Div(
                        [
                            html.Label("Select Country:", style={"fontSize": "18px", "fontWeight": "bold"}),
                            dcc.Dropdown(
                                id="country-dropdown",
                                options=[{"label": country, "value": country} for country in countries],
                                value=countries[0] if countries else None,
                                placeholder="Select a country",
                                style={"width": "160%", "fontSize": "14px", "padding": "8px", "height": "50px"},
                            ),
                        ],
                        style={"marginBottom": "30px", 'textAlign': 'left'}
                    ),

                    # Dropdown for selecting model
                    html.Div(
                        [
                            html.Label("Select Model:", style={"fontSize": "18px", "fontWeight": "bold"}),
                            dcc.Dropdown(
                                id="model-dropdown",
                                options=[{"label": model, "value": model} for model in models],
                                value=models[0] if models else None,
                                placeholder="Select a model",
                                style={"width": "167%", "fontSize": "14px", "padding": "14px", "height": "50px"},
                            ),
                        ],
                        style={"marginBottom": "30px", 'textAlign': 'left'}
                    ),
                ],
                style={
                    "width": "27%",
                    "padding": "20px",
                    "boxShadow": "0 0 10px rgba(0,0,0,0.1)",
                    "borderRadius": "10px",
                    "backgroundColor": "white",
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "flex-start",
                    "height": "95vh",
                }
            ),

            # Right-side container for the plots
            html.Div(
                [
                    # Enhanced descriptive text
                    html.P(
                        "To evaluate the models, a backtesting methodology is applied to each retrained model. The performance is assessed over three cycles, with relevant metrics such as Accuracy and Bias calculated for each cycle and future lag. This comprehensive overview allows for a detailed comparison of model performance across different time horizons. Additionally, the table at the top of the left panel highlights the best-fit model for each country, selected based on the highest Accuracy and lowest Bias across the cycles.",
                        style={
                            "fontSize": "18px",
                            "lineHeight": "1.8",
                            "marginTop": "30px",
                            "marginBottom": "30px",
                            "textAlign": "justify",
                        },
                    ),
                    # Container for the accuracy chart
                    html.Div(
                        dcc.Graph(id="accuracy-chart", config={"displayModeBar": True}),
                        style={"marginBottom": "20px"}
                    ),
                    # Container for the bias chart
                    html.Div(
                        dcc.Graph(id="bias-chart", config={"displayModeBar": True}),
                        style={"marginBottom": "20px"}
                    ),
                    # New plot added here
                    html.Div(
                        dcc.Graph(id="new-plot", config={"displayModeBar": True}),
                        style={"width": "100%", "height": "40vh"}
                    ),
                ],
                style={
                    "width": "75%",
                    "padding": "20px",
                    "boxShadow": "0 0 10px rgba(0,0,0,0.1)",
                    "borderRadius": "10px",
                    "backgroundColor": "white",
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "center",
                    "height": "95vh",
                    "overflowY": "auto",
                }
            ),
        ],
        style={
            "display": "flex",
            "flexDirection": "row",
            "height": "100vh",
            "padding": "20px",
            "gap": "20px",
            "backgroundColor": "#f9f9f9",
        }
    )

   





















