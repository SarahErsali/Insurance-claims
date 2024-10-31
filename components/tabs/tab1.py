from dash import html, dcc, dash_table

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
                children=html.Button('Upload Files', style={'cursor': 'pointer'}),
                multiple=True
            ),
            html.Div(id='upload-status', style={'marginTop': '10px'})
        ], style={'textAlign': 'left', 'paddingLeft': '0px'}),

        # Data table to display the processed data
        html.Div(id='processed-data-table', style={'marginTop': '20px'}),
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
