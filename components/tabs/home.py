from dash import html

def render_home():
    return html.Div([
        # EIOPA Overview
        html.Div([
            html.H3("European Insurance and Occupational Pension Authority (EIOPA)", style={
                'marginBottom': '10px', 
                'fontSize': '32px', 
                'textAlign': 'left',  # Align title to the left
            }),
            html.P(
                "The data for this project originates from the annual European insurance overview published by EIOPA. "
                "The project focuses on the European insurance life sector and is based on annually reported Solvency II information. "
                "This ensures that the dataset provides comprehensive coverage across all countries in the European Economic Area (EEA), "
                "with consistent and reliable reporting standards. In this project, the key goal is to predict net claims incurred through developing predictive models.",
                style={'lineHeight': '2', 'marginBottom': '30px', 'textAlign': 'justify'}  # Justify paragraph text
            )
        ]),

        # Model Developed
        html.Div([
            html.H3("Model Developed", style={
                'marginBottom': '10px', 
                'fontSize': '32px', 
                'textAlign': 'left',  # Align title to the left
            }),
            html.P(
                "For this project, a combination of advanced machine learning models, including XGBoost and LightGBM, and time series models, such as ARIMA and Moving Average, "
                "were meticulously developed. These models are designed to accurately predict net claims incurred by leveraging a diverse range of features, ensuring robust "
                "and insightful predictions.",
                style={'lineHeight': '2', 'marginBottom': '30px', 'textAlign': 'justify'}  # Justify paragraph text
            )
        ]),

        # # Data Summary
        # html.Div([
        #     html.H3("Data Summary", style={
        #         'marginBottom': '10px', 
        #         'fontSize': '32px', 
        #         'textAlign': 'left',  # Align title to the left
        #     }),
        #     html.Ul([
        #         html.Li([html.B("Features:"), " A wide range of features were used, including underwriting risk, number of policies, expenses, etc."]),
        #         html.Li([html.B("Economic Data:"), " This includes key macroeconomic indicators that impact claims behavior, "
        #                 "such as GDP growth rate, inflation rate, unemployment rate, interest rate, and equity return."]),
        #         html.Li([html.B("Crisis Periods:"), " The dataset includes multiple crisis periods like the 2008 Financial "
        #                 "Crisis, the European Debt Crisis, and the COVID-19 pandemic. Each of these periods has a "
        #                 "significant impact on the model's predictions, as they affect economic indicators and insurance "
        #                 "risk."]),
        #         html.Li([html.B("Natural Disaster:"), " Several storm periods in the dataset typically cause a spike in property insurance claims, impacting the risk evaluation."])
        #     ], style={'lineHeight': '2', 'marginBottom': '30px', 'textAlign': 'justify'})  # Justify bullet text
        # ]),

        # Impact on Business Model
        html.Div([
            html.H3("Impact on Analysis and Business Model", style={
                'marginBottom': '10px', 
                'fontSize': '32px', 
                'textAlign': 'left',  # Align title to the left
            }),
            html.P(
                "The features and factors used in the models play a critical role in understanding how external "
                "factors, such as the economy and catastrophic events, influence the number of claims. This aids in "
                "refining the insurance pricing strategy and improving risk management practices.",
                style={'lineHeight': '2', 'marginBottom': '30px', 'textAlign': 'justify'}  # Justify paragraph text
            )
        ])
    ], style={
        'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'alignItems': 'flex-start',
        'paddingTop': '2cm', 'paddingLeft': '3cm', 'paddingRight': '3cm', 'paddingBottom': '3cm',
        'maxWidth': '90%', 'marginLeft': 'auto', 'marginRight': 'auto', 'lineHeight': '1.8',
        'height': 'auto'
    })
