from dash import html

def render_home():
    return html.Div([
        # Introduction Text
        html.P(
            "At BaNex Consulting, we empower the insurance industry to excel in today’s fast-paced world through "
            "cutting-edge, data-driven solutions that drive innovation, efficiency, and growth.",
            style={
                'textAlign': 'center', 'fontSize': '28px', 'marginTop': '2vh', 'lineHeight': '1.9',
                'maxWidth': '80%', 'marginLeft': 'auto', 'marginRight': 'auto', 'paddingLeft': '3cm',
                'paddingRight': '3cm'
            }
        ),

        # Business Problem
        html.H3("Business Problem", style={'marginBottom': '20px', 'fontSize': '32px'}),
        html.P(
            "The project focuses on the property line of business within the insurance industry. "
            "The key goal is to predict Solvency capital requirement (SCR) through developing predictive models for "
            "insurance claims incurred, factoring in various economic indicators, crisis periods, and natural disasters' "
            "impact to improve business decisions.",
            style={'lineHeight': '2', 'marginBottom': '30px'}
        ),

        # Model Developed
        html.H3("Model Developed", style={'marginBottom': '20px', 'fontSize': '32px'}),
        html.P(
            "For this project, advanced machine learning models such as XGBoost and LightGBM were developed. "
            "These models help predict claims incurred based on a variety of features, including economic data, "
            "insurance data, and specific factors affecting the property insurance line.",
            style={'lineHeight': '2', 'marginBottom': '30px'}
        ),

        # Data Summary
        html.H3("Data Summary", style={'marginBottom': '20px', 'fontSize': '32px'}),
        html.Ul([
            html.Li([html.B("Features:"), " A wide range of features were used, including underwriting risk, number of policies, expenses, etc."]),
            html.Li([html.B("Economic Data:"), " This includes key macroeconomic indicators that impact claims behavior, "
                    "such as GDP growth rate, inflation rate, unemployment rate, interest rate, and equity return."]),
            html.Li([html.B("Crisis Periods:"), " The dataset includes multiple crisis periods like the 2008 Financial "
                    "Crisis, the European Debt Crisis, and the COVID-19 pandemic. Each of these periods has a "
                    "significant impact on the model's predictions, as they affect economic indicators and insurance "
                    "risk."]),
            html.Li([html.B("Natural Disaster:"), " Several storm periods in the dataset typically cause a spike in property insurance claims, impacting the risk evaluation."])
        ], style={'lineHeight': '2', 'marginBottom': '30px'}),

        # Impact on Business Model
        html.H3("Impact on Analysis and Business Model", style={'marginBottom': '20px', 'fontSize': '32px'}),
        html.P(
            "The features and factors used in the models play a critical role in understanding how external "
            "factors, such as the economy and catastrophic events, influence the number of claims. This aids in "
            "refining the insurance pricing strategy and improving risk management practices.",
            style={'lineHeight': '2', 'marginBottom': '30px'}
        )
    ], style={
        'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'alignItems': 'center',
        'paddingTop': '2cm', 'paddingLeft': '3cm', 'paddingRight': '3cm', 'paddingBottom': '3cm',
        'textAlign': 'left', 'maxWidth': '90%', 'marginLeft': 'auto', 'marginRight': 'auto', 'lineHeight': '1.8',
        'height': 'auto'
    })
