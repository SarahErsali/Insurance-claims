from dash import html, dcc


def render_tab1():
    return html.Div([

        # Instructional Text
        html.P(
            "To proceed with your data cleaning and analyzing your data as well as developing machine learning models "
            "(e.g., XGBoost and LightGBM) and also time series models (e.g., ARIMA and Moving Average), upload your "
            "database below. You also have the option to upload multiple files. Please bear in mind that only CSV file format is acceptable.",
            style={'textAlign': 'center', 'fontSize': '18px', 'marginBottom': '20px'}
        ),

        # File Upload Section
        html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Button('Upload Files', style={'cursor': 'pointer'}),
                multiple=True
            ),
            html.Div(id='upload-status', style={'marginTop': '10px'})
        ], style={'textAlign': 'center', 'padding': '20px'}),

        html.Div(id='data-overview', style={'marginTop': '20px'})
    ], style={
        'paddingTop': '2cm',
        'paddingLeft': '3cm',
        'paddingRight': '3cm',
        'paddingBottom': '3cm',
        'textAlign': 'center',
        'maxWidth': '90%',
        'marginLeft': 'auto',
        'marginRight': 'auto'
    })

