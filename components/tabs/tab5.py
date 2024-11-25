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
            html.H3(
                "Life LOB",
                style={
                    "textAlign": "center",
                    "marginBottom": "10px",
                    "fontSize": "24px",
                    "fontWeight": "bold",
                    "marginTop": "40px",  # Adjust to bring it below the tabs
                },
            ),
            dash_table.DataTable(
                id='best-models-table',
                columns=[
                    {"name": "Country", "id": "Country"},
                    {"name": "Best Fit Model for Life LOB Data", "id": "Best Fit Model"}
                ],
                data=df.to_dict("records"),
                style_table={
                    'width': '80%',
                    'margin': '0 auto',
                    'marginTop': '20px',  # Add spacing below the title
                },
                style_cell={
                    'textAlign': 'left',
                    'padding': '5px',
                    'fontFamily': 'Arial',
                    'fontSize': '14px'  # Adjust font size to make it compact
                },
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold',
                    'textAlign': 'center'
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
                    "marginTop": "20px",
                    "padding": "10px 20px",
                    "fontSize": "16px",
                    "cursor": "pointer",
                },
            ),
            dcc.Download(id="download-table"),
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "center",
            "justifyContent": "center",
            "height": "70vh",  # Adjust height for download button spacing
        }
    )
    return table
