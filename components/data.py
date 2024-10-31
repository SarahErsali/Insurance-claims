from io import StringIO
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder



#------------ Function for date conversion within the data cleaning step -------------------

# it’s a helper function that’s called within the main function (load_and_process_uploaded_data). 
# Defining it first ensures it’s available whenever needed within the main function, regardless of when user data is provided.
# It’s just a matter of organization, not execution timing.

def convert_quarter_to_date(quarter_str):
    year, quarter = quarter_str.split()
    year = int(year)
    quarter_mapping = {'Q1': 1, 'Q2': 4, 'Q3': 7, 'Q4': 10}
    return pd.Timestamp(year=year, month=quarter_mapping[quarter], day=1)

def load_and_process_uploaded_data(contents, filenames, existing_dataframes):
    for content, filename in zip(contents, filenames):
        country_name = filename.split('.')[0]
        content_type, content_string = content.split(',')
        decoded = pd.read_csv(StringIO(content_string))

        if country_name in existing_dataframes:
            existing_dataframes[f"{country_name}_new"] = decoded
        else:
            existing_dataframes[country_name] = decoded

    for name, df in existing_dataframes.items():
        df.columns = df.columns.str.strip()

        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_numeric(df[col].str.replace(' ', '').str.replace(',', ''), errors='coerce')
            except Exception:
                continue

        for col in df.columns:
            if 'quarter' in col.lower() or 'date' in col.lower():
                df[col] = df[col].apply(lambda x: convert_quarter_to_date(x) if isinstance(x, str) else x)
            # Ensure 'Date' is in datetime format
            if col.lower() == 'date':
                df[col] = pd.to_datetime(df[col], errors='coerce')

        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())

    combined_df = pd.DataFrame()
    for country, df in existing_dataframes.items():
        df['Country'] = country
        combined_df = pd.concat([combined_df, df], axis=0, ignore_index=True)

    le = LabelEncoder()
    combined_df['Country'] = le.fit_transform(combined_df['Country'])

    numeric_columns = combined_df.select_dtypes(include=['float64', 'int64']).columns
    lagged_columns = []
    for col in numeric_columns:
        for lag in range(1, 9):
            lagged_col = combined_df.groupby('Country')[col].shift(lag)
            lagged_columns.append(lagged_col.rename(f"{col}_Lag{lag}"))

    # Concatenate all lagged columns at once to avoid fragmentation
    combined_df = pd.concat([combined_df] + lagged_columns, axis=1)

    if 'Date' in combined_df.columns:
        combined_df['Year'] = combined_df['Date'].dt.year
        combined_df['Quarter'] = combined_df['Date'].dt.quarter

    combined_df = combined_df.reset_index(drop=True)
    combined_df = combined_df.groupby('Country', as_index=False).apply(lambda group: group.fillna(method='ffill').fillna(method='bfill'))
    combined_df = combined_df.reset_index(drop=True)

    return combined_df








# ### Note: The XGBoost library does not support datetime64[ns] data types, therefore Date has to be dropped from X

# # Training set: 2008-2019
# train_data = property_data_feature_selected[(property_data_feature_selected['Date'].dt.year >= 2008) & (property_data_feature_selected['Date'].dt.year <= 2019)]

# # Validation set: 2020-2022
# val_data = property_data_feature_selected[(property_data_feature_selected['Date'].dt.year >= 2020) & (property_data_feature_selected['Date'].dt.year <= 2022)]

# # Blind test set: 2023-2024
# blind_test_data = property_data_feature_selected[(property_data_feature_selected['Date'].dt.year >= 2023) & (property_data_feature_selected['Date'].dt.year <= 2024)]

# X_train = train_data.drop(['Claims_Incurred', 'Date'], axis=1, errors='ignore') 
# y_train = train_data['Claims_Incurred']

# X_val = val_data.drop(['Claims_Incurred', 'Date'], axis=1, errors='ignore')
# y_val = val_data['Claims_Incurred']

# X_blind_test = blind_test_data.drop(['Claims_Incurred', 'Date'], axis=1, errors='ignore')
# y_blind_test = blind_test_data['Claims_Incurred']
# y_blind_test.index = blind_test_data['Date']  # This ensures that y_blind_test has a datetime index instead of an integer index

# # Combine Training and Validation Sets
# X_combined = pd.concat([X_train, X_val], ignore_index=True)
# y_combined = pd.concat([y_train, y_val], ignore_index=True)

# Export the variables
#__all__ = ['load_uploaded_data']