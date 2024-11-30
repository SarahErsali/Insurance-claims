from dash import dash_table, dcc, html
import pandas as pd

def generate_best_model_table(results):
    """
    Generate a Dash DataTable for the best-fit models per country with a download button.

    Args:
        results (dict): The results dictionary containing the best models.

    Returns:
        html.Div: Dash HTML layout containing the table and download button.
    """
    best_models = results.get("best_models", {})
    predictions = results.get("ml_predictions", {}).get("all_countries_predictions", {}) # check dictionary
    
    # Prepare the data for the table
    data = [
        {
            "Country": country,
            "Best Fit Model": model_info["model"],
            "Prediction Values": ", ".join(map(str, predictions.get(model_info["model"], [])))  # Format predictions as a comma-separated string
        }
        for country, model_info in best_models.items()
    ]

    # Convert to a DataFrame for sorting (optional)
    df = pd.DataFrame(data)
    df = df.sort_values(by="Country").reset_index(drop=True)

    # Generate Dash DataTable with a download button
    layout = html.Div(
        [
            # Instructional paragraph with top spacing
            html.P(
                "Below is the summary of the best-performing models, including XGBoost, LightGBM, ARIMA, and Moving Average, "
                "selected for each country. These models were evaluated and chosen based on their ability to provide accurate predictions for the dataset. "
                "You can download the results for further analysis using the button below.",
                style={
                    "lineHeight": "1.8",  # Consistent line spacing
                    "marginBottom": "20px",  # Spacing below the paragraph
                    "textAlign": "justify",  # Justified text
                    "marginTop": "2cm",  # Space from the top of the page
                    "fontSize": "18px",  # Font size for readability
                    "paddingLeft": "5cm",  # Padding on the left for alignment
                    "paddingRight": "5cm",  # Padding on the right for alignment
                }
            ),
            # Download Button
            html.Div(
                [
                    html.Button(
                        "Download Table",
                        id="download-button",
                        style={
                            "marginTop": "20px",
                            "padding": "15px 30px",  # Increased padding for larger button size
                            "fontSize": "18px",  # Increased font size for better readability
                            "fontWeight": "bold",  # Bold for emphasis
                            "backgroundColor": "#007BFF",  # Bootstrap primary blue
                            "color": "white",  # White text for contrast
                            "border": "none",  # Remove border
                            "borderRadius": "5px",  # Rounded corners
                            "cursor": "pointer",  # Pointer cursor for interactivity
                            "textAlign": "center",  # Center text in the button
                            "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.1)",  # Add subtle shadow
                        },
                    ),
                    dcc.Download(id="download-table"),  # Add the download component
                ],
                style={
                    "textAlign": "left",  # align button within the div
                    "marginBottom": "50px", 
                    "paddingLeft": "2cm",  # Align the button with the paragraph
                    "paddingRight": "27.0cm"  # Align the button with the paragraph
                }
            ),
            # Table placed below the button
            html.Div(
                dash_table.DataTable(
                    id='best-models-table',
                    columns=[
                        {"name": "Country", "id": "Country"},
                        {"name": "Best Fit Model for Life LOB", "id": "Best Fit Model"},
                        {"name": "Prediction Values", "id": "Prediction Values"}
                    ],
                    data=df.to_dict("records"),
                    style_table={
                        'width': '80%',
                        'margin': '0 auto',  # Center table with top and bottom margin
                    },
                    style_cell={
                        'textAlign': 'center',  # Center-align text in the table
                        'padding': '10px',  # Increased padding for readability
                        'fontFamily': 'Arial',
                        'fontSize': '14px',  # Adjust font size to make it compact
                    },
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'lineHeight': '1.8',  # Consistent line height
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ]
                ),
                style={"width": "60%", "margin": "0 auto"}  # Center the table
            ),
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "center",  # Center-align all child components, including the table
            "justifyContent": "flex-start",  # Align content to the top
            "height": "auto",  # Adjust height for download button spacing
            "textAlign": "justify",  # Align text for consistency
            "lineHeight": "1.8",  # Consistent line height
            "paddingBottom": "3cm",  # Add space at the bottom of the page
        }
    )
    return layout
