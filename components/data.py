from io import StringIO
import pandas as pd
import numpy as np
from datetime import datetime



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
    except Exception as e:
        print(f"Conversion error with value: {quarter_str} -> {e}")
        return pd.NaT  # Return NaT for invalid formats
    

def load_and_process_uploaded_data(contents, filenames, existing_dataframes):
        
    for name, df in existing_dataframes.items():
        df.columns = df.columns.str.strip()
        #print("Columns for country:", name, df.columns)
        
        for col in df.select_dtypes(include='object').columns:
            if 'date' not in col.lower(): 
                try:
                    df[col] = pd.to_numeric(df[col].str.replace(' ', '').str.replace(',', ''), errors='coerce')
                except Exception:
                    continue

        # Apply the quarter-to-date conversion
        if 'Date' in df.columns.tolist():
            #print(f"Before conversion, 'Date' sample values in {name}: {df['Date'].head()}")
            df['Date'] = df['Date'].apply(lambda x: convert_quarter_to_date(x) if isinstance(x, str) else x)
            #print(f"After conversion, 'Date' sample values in {name}: {df['Date'].head()}")
        else:
            print(f"No 'Date' column found in {name}.")

        # Fill NaNs with mean for numeric columns
        for col in df.select_dtypes(include='float64').columns:
            if df[col].isnull().any():
                #print(f"Filling NaNs in column {col} with mean for {name}")
                df[col] = df[col].fillna(df[col].mean())


    combined_df = pd.DataFrame()
    for country, df in existing_dataframes.items():
        df['Country'] = country
        combined_df = pd.concat([combined_df, df], axis=0) 


    numeric_columns = combined_df.select_dtypes(include='float64').columns 

    for col in numeric_columns:
        for lag in range(1, 9):
            combined_df[f'{col}_Lag{lag}'] = combined_df.groupby('Country')[col].shift(lag)
            

    # Extract Year and Quarter if 'Date' is properly formatted
    if 'Date' in combined_df.columns and pd.api.types.is_datetime64_any_dtype(combined_df['Date']):
        combined_df['Year'] = combined_df['Date'].dt.year
        combined_df['Quarter'] = combined_df['Date'].dt.quarter
    else:
        print("Warning: 'Date' column not in expected datetime format after conversion.")
        combined_df['Year'] = np.nan
        combined_df['Quarter'] = np.nan

    combined_df = combined_df.groupby('Country').apply(lambda group: group.ffill().bfill()).reset_index(drop=True)
    # Debugging after processing
    #print(f"Processed combined_df size: {combined_df.shape}")  #Processed combined_df size: (31, 67)

    # Debug check for any remaining NaNs after processing
    if combined_df.isnull().any().any():
        print("Warning: NaNs detected in combined_df after preprocessing.")
        print(combined_df.isnull().sum())

    # # Add specific check for the target column
    # if combined_df['NET Claims Incurred'].isnull().any():
    #     print("Warning: NaNs detected in target column after preprocessing.")

    return combined_df



def prepare_for_arima_ma(combined_df, target_column):
    
    # Extract only the Date and target columns
    if 'Date' not in combined_df.columns or target_column not in combined_df.columns:
        raise ValueError("The combined_df must contain 'Date' and target_column for ARIMA/MA modeling.")
    
    print("\nStarting ARIMA/MA preprocessing")

    # Extract necessary columns
    arima_df = combined_df[['Date', target_column]].copy()
    #print(f"Initial arima_df shape: {arima_df.shape}")
    #print(f"Initial arima_df head:\n{arima_df.head()}")

    # Drop rows with NaN in the target column
    arima_df = arima_df.dropna(subset=[target_column])
    #print(f"Shape after dropping NaNs in target column: {arima_df.shape}")
    #print(f"Sample of arima_df after dropping NaNs:\n{arima_df.head()}")    

    # Set Date as index (optional, based on ARIMA/MA requirements)
    arima_df.set_index('Date', inplace=True)
    #print(f"arima_df after setting 'Date' as index:\n{arima_df.head()}")
    #print(f"arima_df index type: {type(arima_df.index)}")

    # Explicitly set the frequency for the index
    if not arima_df.index.freq:
        arima_df.index = pd.date_range(start=arima_df.index.min(), periods=len(arima_df), freq="QS")
        print("Frequency set for arima_df index.")
    else:
        print("Frequency already set for arima_df index.")

    # Check for missing periods
    missing_periods = pd.date_range(start=arima_df.index.min(), end=arima_df.index.max(), freq="QS").difference(arima_df.index)
    if len(missing_periods) > 0:
        print(f"Missing periods in the time series: {missing_periods}")
    else:
        print("No missing periods in the time series.")

    # Final check
    #print(f"Final arima_df shape: {arima_df.shape}")
    #print(f"Final arima_df head:\n{arima_df.head()}")
    #print(f"arima_df index frequency: {arima_df.index.freq}")
    #print(f"ARIMA dataset size: {arima_df.shape[0]}")  #ARIMA dataset size: 31

    return arima_df




# def load_and_process_uploaded_data(contents, filenames, existing_dataframes):
#     combined_df = pd.DataFrame()  # Initialize an empty DataFrame to store processed data
    
#     for country, df in existing_dataframes.items():
#         df.columns = df.columns.str.strip()  # Clean column names
#         print(f"Processing data for country: {country}")
        
#         # Convert object columns to numeric where possible
#         for col in df.select_dtypes(include='object').columns:
#             if 'date' not in col.lower(): 
#                 try:
#                     df[col] = pd.to_numeric(df[col].str.replace(' ', '').str.replace(',', ''), errors='coerce')
#                 except Exception as e:
#                     print(f"Error converting column {col} in {country}: {e}")
#                     continue
        
#         # Convert 'YYYY Qn' format to datetime
#         if 'Date' in df.columns:
#             df['Date'] = df['Date'].apply(lambda x: convert_quarter_to_date(x) if isinstance(x, str) else x)
#             print(f"Sample 'Date' values after conversion for {country}: {df['Date'].head()}")
#         else:
#             print(f"Warning: No 'Date' column found in {country}.")
#             continue  # Skip further processing if 'Date' is missing

#         # Fill NaNs for numeric columns with the column mean
#         for col in df.select_dtypes(include='float64').columns:
#             if df[col].isnull().any():
#                 print(f"Filling NaNs in column '{col}' with mean for {country}.")
#                 df[col] = df[col].fillna(df[col].mean())
        
#         # Generate lagged features specific to this country
#         numeric_columns = df.select_dtypes(include='float64').columns
#         for col in numeric_columns:
#             for lag in range(1, 9):
#                 df[f'{col}_Lag{lag}'] = df[col].shift(lag)
        
#         # Fill remaining NaNs with forward/backward filling after creating lagged features
#         df = df.ffill().bfill()
        
#         # Add a 'Country' column and append the cleaned data to combined_df
#         df['Country'] = country
#         combined_df = pd.concat([combined_df, df], axis=0)

#     # Extract Year and Quarter from the 'Date' column
#     if 'Date' in combined_df.columns and pd.api.types.is_datetime64_any_dtype(combined_df['Date']):
#         combined_df['Year'] = combined_df['Date'].dt.year
#         combined_df['Quarter'] = combined_df['Date'].dt.quarter
#     else:
#         print("Warning: 'Date' column not in expected datetime format.")
#         combined_df['Year'] = np.nan
#         combined_df['Quarter'] = np.nan

#     # Debug checks for remaining NaNs
#     if combined_df.isnull().any().any():
#         print("Warning: NaNs detected in combined_df after preprocessing.")
#         print(combined_df.isnull().sum())

#     # Check for NaNs in the target column
#     if combined_df['NET Claims Incurred'].isnull().any():
#         print("Warning: NaNs detected in the target column after preprocessing.")

#     return combined_df

