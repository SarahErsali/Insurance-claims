from dash import dash_table, dcc, html
import pandas as pd

def generate_best_model_table(results):
    """
    Generate a Dash layout for the best-fit models per country with side-by-side containers and borders.
    """
    best_models = results.get("best_models", {})
    quarters = results["prediction_periods"]
    #---------
    # Extract country options for the dropdown
    #country_options = [{"label": country, "value": country} for country in best_models.keys()]
    #-----------
    # Prepare data for the table
    data = [
        {
            "Country": country if i == 0 else "",
            "Best Fit Model for Life LOB": model_info["model"] if i == 0 else "",
            "Year / Quarter": quarters[i],
            "Prediction Values": model_info.get('predictions', [])[i]
            #"Prediction Values": "<br>".join([f"{q}: {p}" for q, p in zip(quarters, map(str, model_info.get('predictions', [])))])
              #join()
        }
        for country, model_info in best_models.items()
        for i in range(len(quarters))
    ]

    # Convert to DataFrame
    df = pd.DataFrame(data).sort_values(by="Country").reset_index(drop=True)

    layout = html.Div([
        html.Div([  # Left Container

            # Download Button
            html.Div([
                html.Button(
                    "Download Table",
                    id="download-button",
                    style={
                        "padding": "15px 30px",
                        "fontSize": "18px",
                        "fontWeight": "bold",
                        "backgroundColor": "#007BFF",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "5px",
                        "cursor": "pointer",
                        "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.1)"
                    }
                ),
                dcc.Download(id="download-table"),
            ], style={"textAlign": "center", "width": "100%", "marginTop": "50px"})
        ], style={
            "width": "24%",
            "padding": "15px",
            "boxShadow": "0 0 10px rgba(0,0,0,0.1)",
            "backgroundColor": "white",
            "borderRadius": "10px",
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "flex-start",  # Align all contents to the left
            "height": "95vh"
        }),

        # Right Container
        html.Div([
            html.P(
                "Below is the summary of the best-performing models, including relevant prediction values, "
                "selected for each country. These models were evaluated and chosen based on their ability to "
                "provide accurate predictions for the dataset.",
                style={
                    "lineHeight": "1.8",
                    "marginBottom": "20px",
                    "textAlign": "justify",
                    "fontSize": "16px"
                }
            ),
            dash_table.DataTable(
                id="best-models-table",
                columns=[
                    {"name": "Country", "id": "Country"},
                    {"name": "Best Fit Model for Life LOB", "id": "Best Fit Model for Life LOB"},
                    {"name": "Year / Quarter", "id": "Year / Quarter"},
                    {"name": "Prediction Values", "id": "Prediction Values"},
                ],
                data=data,  # Use the adjusted data
                style_table={"width": "70%", "margin": "0 auto", "overflowX": "auto",},
                style_cell={
                    "textAlign": "center",
                    "padding": "8px",
                    "fontFamily": "Arial",
                    "fontSize": "14px",
                    "borderLeft": "3px solid black",  # Set thick vertical borders on the left
                    "borderRight": "3px solid black",  # Set thick vertical borders on the right
                },
                style_header={
                    "backgroundColor": "rgb(230, 230, 230)",
                    "fontWeight": "bold",
                    "borderLeft": "3px solid black",  # Match vertical border thickness for header
                    "borderRight": "3px solid black",  # Match vertical border thickness for header
                    "borderTop": "3px solid black",   # Add a thick top border for the header row
                    "borderBottom": "3px solid black",  # Add a thick bottom border for the header row
                },
                merge_duplicate_headers=True,  # Optional: To group headers for a cleaner look
                style_data_conditional=[
                    {
                        "if": {"row_index": row_index},  # Apply style to every 4th row (excluding header)
                        "borderBottom": "3px solid black",  # Thicker bottom border
                    }
                    for row_index in range(3, len(data), 4)  # Starting from the 4th row, apply to every 4th row
                ],
            ),
        ], style={
            'width': '75%',
            'padding': '15px',
            'boxShadow': '0 0 10px rgba(0, 0, 0, 0.1)',
            'backgroundColor': 'white',
            'borderRadius': '10px',
            'height': '95vh',
            'overflowY': 'auto'
        })
    ], style={
        'display': 'flex',
        'flexDirection': 'row',
        'justifyContent': 'space-between',
        'padding': '20px',
        'backgroundColor': '#f8f9fa',
        'height': '100vh',
        'gap': '10px'
    })

    return layout


    # layout = html.Div([
    #     html.Div([  # Left Container
    #         html.Div([
    #             html.Button(
    #                 "Download Table",
    #                 id="download-button",
    #                 style={
    #                     "padding": "15px 30px",
    #                     "fontSize": "18px",
    #                     "fontWeight": "bold",
    #                     "backgroundColor": "#007BFF",
    #                     "color": "white",
    #                     "border": "none",
    #                     "borderRadius": "5px",
    #                     "cursor": "pointer",
    #                     "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.1)"
    #                 }
    #             ),
    #             dcc.Download(id="download-table"),
    #         ], style={"textAlign": "center", "marginTop": "20px"}) 
    #     ], style={
    #         "width": "24%", 
    #         'padding': '15px',
    #         'boxShadow': '0 0 10px rgba(0,0,0,0.1)',
    #         'backgroundColor': 'white',
    #         'borderRadius': '10px',
    #         'display': 'flex',
    #         'flexDirection': 'column',
    #         'alignItems': 'center',
    #         'height': '95vh'
            
    #     }),

    #     html.Div([  # Right Container
    #         html.P(
    #             "Below is the summary of the best-performing models, including relevant prediction values, "
    #             "selected for each country. These models were evaluated and chosen based on their ability to "
    #             "provide accurate predictions for the dataset.",
    #             style={
    #                 "lineHeight": "1.8",
    #                 "marginBottom": "20px",
    #                 "textAlign": "justify",
    #                 "fontSize": "16px"
    #             }
    #         ),
    #         dash_table.DataTable(
    #             id='best-models-table',
    #             columns=[
    #                 {"name": "Country", "id": "Country"},
    #                 {"name": "Best Fit Model", "id": "Best Fit Model"},
    #                 {"name": "Prediction Values", "id": "Prediction Values"}
    #             ],
    #             data=df.to_dict("records"),
    #             style_table={'width': '100%'},
    #             style_cell={
    #                 'textAlign': 'center',
    #                 'padding': '8px', # reduced padding for compactness
    #                 'fontFamily': 'Arial',
    #                 'fontSize': '14px'
    #             },
    #             style_header={
    #                 'backgroundColor': 'rgb(230, 230, 230)',
    #                 'fontWeight': 'bold'
    #             }
    #         ),
    #     ], style={
    #         'width': '75%',
    #         'padding': '15px',
    #         'boxShadow': '0 0 10px rgba(0, 0, 0, 0.1)',
    #         'backgroundColor': 'white',
    #         'borderRadius': '10px',
    #         'height': '95vh',
    #         'overflowY': 'auto'
            
    #     })
    # ], style={
    #     'display': 'flex',
    #     'flexDirection': 'row',
    #     'justifyContent': 'space-between', # Keeps alignment tight
    #     'padding': '20px',
    #     'backgroundColor': '#f8f9fa',
    #     'height': '100vh',
    #     'gap': '10px'  # Reduced gap between containers
        
    # })

    # return layout





