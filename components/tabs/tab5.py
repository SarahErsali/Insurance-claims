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
    
    # Prepare the data for the table
    data = [{"Country": country, "Best Fit Model": model_info["model"]} for country, model_info in best_models.items()]

    # Convert to a DataFrame for sorting (optional)
    df = pd.DataFrame(data)
    df = df.sort_values(by="Country").reset_index(drop=True)

    # Generate Dash DataTable with a download button
    table = html.Div(
        [
            # Enhanced paragraph with top spacing
            html.P(
                "Below is the summary of the best-performing models, including XGBoost, LightGBM, ARIMA, and Moving Average, "
                "selected for each country. These models were evaluated and chosen based on their ability to provide accurate predictions for the dataset. "
                "You can download the results for further analysis using the button below.",
                style={
                    "lineHeight": "1.8",  # Consistent line spacing
                    "marginBottom": "30px",  # Spacing below the paragraph
                    "textAlign": "justify",  # Justified text
                    "marginTop": "2cm",  # Increased space from the top of the page
                    "fontSize": "16px",  # Font size for readability
                    "paddingLeft": "5cm",  # Correct padding for left
                    "paddingRight": "5cm",  # Correct padding for right
                }
            ),
            dash_table.DataTable(
                id='best-models-table',
                columns=[
                    {"name": "Country", "id": "Country"},
                    {"name": "Best Fit Model for Life LOB", "id": "Best Fit Model"}
                ],
                data=df.to_dict("records"),
                style_table={
                    'width': '80%',
                    'margin': '0 auto',
                    'marginTop': '20px',  # Add spacing below the title
                },
                style_cell={
                    'textAlign': 'left',
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
            html.Button(
                "Download Table",
                id="download-button",
                style={
                    "marginTop": "40px",
                    "padding": "15px 30px",  # Increased padding for larger button size
                    "fontSize": "18px",  # Increased font size for better readability
                    "fontWeight": "bold",  # Bold for emphasis
                    "backgroundColor": "#007BFF",  # Bootstrap primary blue
                    "color": "white",  # White text for contrast
                    "border": "none",  # Remove border
                    "borderRadius": "5px",  # Rounded corners
                    "cursor": "pointer",  # Pointer cursor for interactivity
                    "textAlign": "center",  # Center text in the button
                    "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.1)"  # Add subtle shadow
                },
            ),
            dcc.Download(id="download-table"),
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "center",
            "justifyContent": "center",
            "height": "auto",  # Adjust height for download button spacing
            "textAlign": "justify",  # Align text for consistency
            "lineHeight": "1.8",  # Consistent line height
            "paddingBottom": "3cm",  # Add space at the bottom of the page
        }
    )
    return table
