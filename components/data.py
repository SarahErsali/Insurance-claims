import pandas as pd
import numpy as np
import os
from io import StringIO
from sklearn.preprocessing import LabelEncoder


# ---------------------- Loading Database -------------------------------



def load_uploaded_data(contents, filename, existing_dataframes):
    # Use filename without extension as the key
    country_name = filename.split('.')[0]
    
    # Decode the uploaded file contents
    content_type, content_string = contents.split(',')
    decoded = pd.read_csv(StringIO(content_string))

    # Add to the dictionary, ensuring no overwriting of keys
    if country_name in existing_dataframes:
        existing_dataframes[country_name + '_new'] = decoded
    else:
        existing_dataframes[country_name] = decoded
    
    return existing_dataframes


#---------------- Data Cleaning ------------------------------------------


# Convert 'YYYY Qn' format into datetime
def convert_quarter_to_date(quarter_str):
    try:
        year, quarter = quarter_str.split()
        year = int(year)
        quarter_mapping = {
            'Q1': 1,  # January
            'Q2': 4,  # April
            'Q3': 7,  # July
            'Q4': 10  # October
        }
        return pd.Timestamp(year=year, month=quarter_mapping[quarter], day=1)
    except Exception as e:
        return np.nan

# Generalized Data Cleaning Function
def clean_data(df):
    # 1. Strip leading and trailing spaces from column names
    df.columns = df.columns.str.strip()

    # 2. Identify numeric columns and clean formatting issues
    for col in df.select_dtypes(include=['object']).columns:
        try:
            # Remove spaces, commas, or any other formatting and convert to numeric
            df[col] = pd.to_numeric(df[col].str.replace(' ', '').str.replace(',', ''), errors='coerce')
        except Exception as e:
            pass

    # 3. Detect and convert 'quarter' columns to datetime
    for col in df.columns:
        if 'quarter' in col.lower() or 'date' in col.lower():
            df[col] = df[col].apply(lambda x: convert_quarter_to_date(x) if isinstance(x, str) else x)

    # 4. Impute missing values in numeric columns with the column mean
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)
    
    return df

# Bulk cleaning function for all uploaded dataframes
def clean_uploaded_dataframes(uploaded_data):
    cleaned_data = {}
    for name, df in uploaded_data.items():
        cleaned_data[name] = clean_data(df)
    return cleaned_data



#---------------- Feature Engineering ------------------------------------------



import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def combine_and_process_data(dataframes, max_lags=8, encoding='label'):
    combined_data = pd.DataFrame()

    for country, df in dataframes.items():
        # Ensure the 'Date' column exists and is in datetime format
        if 'Date' not in df.columns:
            print(f"Error: Dataset for {country} is missing a 'Date' column.")
            continue  # Skip this dataset

        # Convert 'Date' to datetime format if it's not already
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Check for any conversion errors (NaT values) and handle them
        if df['Date'].isnull().any():
            print(f"Warning: Invalid date format in {country} dataset; dropping rows with invalid dates.")
            df = df.dropna(subset=['Date'])

        df['Country'] = country
        combined_data = pd.concat([combined_data, df], axis=0)
    print("Data combined successfully.")  # Debugging output

    # Encoding
    if encoding == 'label':
        le = LabelEncoder()
        combined_data['Country'] = le.fit_transform(combined_data['Country'])
        print("Label encoding applied.")  # Debugging output
    elif encoding == 'onehot':
        combined_data = pd.get_dummies(combined_data, columns=['Country'])
        print("One-hot encoding applied.")  # Debugging output

    # Generate lag features
    lag_columns = combined_data.select_dtypes(include=[np.number]).columns  # Use all numerical columns for lagging
    for col in lag_columns:
        for lag in range(1, max_lags + 1):
            combined_data[f'{col}_Lag{lag}'] = combined_data.groupby('Country')[col].shift(lag)
    print("Lag features generated.")  # Debugging output

    # Additional feature engineering: 'Year' and 'Quarter' features from 'Date'
    combined_data['Year'] = combined_data['Date'].dt.year
    combined_data['Quarter'] = combined_data['Date'].dt.quarter
    print("Year and Quarter features added.")  # Debugging output

    # Handle missing values
    combined_data = combined_data.groupby('Country').apply(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    print("Missing values handled.")  # Debugging output

    return combined_data




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