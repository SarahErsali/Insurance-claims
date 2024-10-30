from dash import html, dcc

def render_tab1():
    return html.Div([
        # Instructional Text for Upload
        html.P(
            "To begin data cleaning and analysis, please upload your database file(s) below. You may upload multiple CSV files, which will be used for building machine learning models (e.g., XGBoost, LightGBM) and time series models (e.g., ARIMA, Moving Average). Note that only CSV file formats are accepted.",
            style={'textAlign': 'left', 'fontSize': '18px', 'marginBottom': '20px'}
        ),

        # File Upload Section with adjusted alignment
        html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Button('Upload Files', style={'cursor': 'pointer', 'marginLeft': '0px', 'verticalAlign': 'middle'}),
                multiple=True
            ),
            html.Div(id='upload-status', style={'marginTop': '10px'})
        ], style={'textAlign': 'left', 'paddingLeft': '0px'}),  # Set padding to 0 to align with the Clean Data button

        # Instructional Text for Clean Data Button
        html.P(
            "Once your data is uploaded, you may proceed with data cleaning. Simply click on the 'Clean Data' button below to start the cleaning process for all uploaded files.",
            style={'textAlign': 'left', 'fontSize': '18px', 'marginTop': '40px', 'marginBottom': '20px'}
        ),

        # Clean Data Button and Status
        html.Div([
            html.Button('Clean Data', id='clean-data-button', n_clicks=0, style={'cursor': 'pointer'}),
            html.Div(id='clean-status', style={'marginTop': '10px'})  # Display cleaning status
        ]),

        html.Div(id='data-overview', style={'marginTop': '20px'})
    ], style={
        'paddingTop': '2cm',
        'paddingLeft': '3cm',
        'paddingRight': '3cm',
        'paddingBottom': '3cm',
        'textAlign': 'left',  
        'maxWidth': '90%',
        'marginLeft': 'auto',
        'marginRight': 'auto'
    })
