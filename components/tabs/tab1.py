from dash import html, dcc

def render_tab1():
    return html.Div([
        # Instructional Text for Upload and Processing
        html.P(
            "Please upload your CSV file(s) for processing, cleaning, and feature engineering. The processed data will display below after completion.",
            style={'textAlign': 'left', 'fontSize': '18px', 'marginBottom': '20px'}
        ),

        # File Upload and Process Button in One
        html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Button('Upload and Process Data', style={'cursor': 'pointer'}),
                multiple=True
            ),
            html.Div(id='upload-status', style={'marginTop': '10px'}),
            html.Div(id='final-df-display', style={'marginTop': '20px'})  # Display for the processed DataFrame
        ], style={'textAlign': 'left', 'paddingLeft': '0px'})  
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
