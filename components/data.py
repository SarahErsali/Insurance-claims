from io import StringIO
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder



#------------ Function for date conversion within the data cleaning step -------------------

# it’s a helper function that’s called within the main function (load_and_process_uploaded_data). 
# Defining it first ensures it’s available whenever needed within the main function, regardless of when user data is provided.
# It’s just a matter of organization, not execution timing.


# Function for converting 'YYYY Qn' format to a datetime object
def convert_quarter_to_date(quarter_str):
    try:
        year, quarter = quarter_str.split()
        year = int(year)
        quarter_mapping = {'Q1': 1, 'Q2': 4, 'Q3': 7, 'Q4': 10}
        return pd.Timestamp(year=year, month=quarter_mapping[quarter], day=1)
    except Exception:
        return pd.NaT  # Return NaT for invalid formats

def load_and_process_uploaded_data(contents, filenames, existing_dataframes):
    for content, filename in zip(contents, filenames):
        country_name = filename.split('.')[0]
        content_type, content_string = content.split(',')
        decoded = pd.read_csv(StringIO(content_string))

        # Avoid overwriting existing dataframes with the same country name
        if country_name in existing_dataframes:
            existing_dataframes[f"{country_name}_new"] = decoded
        else:
            existing_dataframes[country_name] = decoded

    for name, df in existing_dataframes.items():
        df.columns = df.columns.str.strip()

        # Convert columns with numeric strings into floats
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_numeric(df[col].str.replace(' ', '').str.replace(',', ''), errors='coerce')
            except Exception:
                continue

        # Apply date conversion on columns that might contain 'quarter' or 'date' in their names
        for col in df.columns:
            if 'quarter' in col.lower() or 'date' in col.lower():
                df[col] = df[col].apply(lambda x: convert_quarter_to_date(x) if isinstance(x, str) else x)
            if col.lower() == 'date':
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Impute missing values for numeric columns
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())

    # Combine all country dataframes into a single dataframe
    combined_df = pd.DataFrame()
    for country, df in existing_dataframes.items():
        df['Country'] = country
        combined_df = pd.concat([combined_df, df], axis=0, ignore_index=True)

    # Encode 'Country' column
    le = LabelEncoder()
    combined_df['Country'] = le.fit_transform(combined_df['Country'])

    # Generate lag features for numeric columns
    numeric_columns = combined_df.select_dtypes(include=['float64', 'int64']).columns
    lagged_dataframes = [combined_df]

    for lag in range(1, 9):
        lagged_df = combined_df[numeric_columns].groupby(combined_df['Country']).shift(lag)
        lagged_df = lagged_df.add_suffix(f"_Lag{lag}")
        lagged_dataframes.append(lagged_df)

    combined_df = pd.concat(lagged_dataframes, axis=1)

    # Extract Year and Quarter from 'Date' if it exists and is in datetime format
    if 'Date' in combined_df.columns and pd.api.types.is_datetime64_any_dtype(combined_df['Date']):
        combined_df['Year'] = combined_df['Date'].dt.year
        combined_df['Quarter'] = combined_df['Date'].dt.quarter
    else:
        combined_df['Year'] = np.nan
        combined_df['Quarter'] = np.nan

    # Forward and backward fill missing values within each group
    combined_df = combined_df.groupby('Country').apply(lambda group: group.ffill().bfill()).reset_index(drop=True)

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