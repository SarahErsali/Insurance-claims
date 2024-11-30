from dash import html, dcc, dash_table

def render_tab1():
    return html.Div([
        # Instructional Text for Upload
        html.P(
            "Start your data cleaning and analysis journey by uploading your database file(s) below. You can upload multiple CSV files, which will serve as the foundation for building advanced machine learning models (such as XGBoost and LightGBM) and time series models (including ARIMA and Moving Average). Please ensure that your files are in CSV format, as only this format is supported for seamless processing.",
            style={
                'lineHeight': '2',  # Line spacing
                'marginBottom': '30px',  # Bottom margin for spacing
                'textAlign': 'justify',  # Justified alignment
                'fontSize': '18px',
            }
        ),

        # File Upload Section
        html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Button(
                    'Upload Files', 
                    style={
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
                    }
                ),
                multiple=True
            ),
            html.Div(id='upload-status', style={'marginTop': '10px', 'fontWeight': 'bold', 'color': 'green'})
        ], style={'textAlign': 'left', 'paddingLeft': '0px'}),

        # Data table to display the processed data
        html.Div(id='processed-data-table', style={'marginTop': '20px'}),

        # Generate results button and status output
        html.Button(
            'Generate Results',
            id='generate-button',
            style={
                "marginTop": "30px",
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
            }
        ),
        html.Div(id='results-generation-status', style={'marginTop': '10px', 'fontWeight': 'bold', 'color': 'green'}),
    ], style={
        'paddingTop': '2cm',
        'paddingLeft': '3cm',
        'paddingRight': '3cm',
        'paddingBottom': '3cm',
        'textAlign': 'left',  
        'maxWidth': '90%',
        'marginLeft': 'auto',
        'marginRight': 'auto',
        'lineHeight': '1.8',  # Adjusted line height for consistency
    })
