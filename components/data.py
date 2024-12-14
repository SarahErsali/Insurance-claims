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
    #print(f"Processed combined_df size: {combined_df.shape}")  #Processed combined_df size: (31, 67)

    # Debug check for any remaining NaNs after processing
    if combined_df.isnull().any().any():
        print("Warning: NaNs detected in combined_df after preprocessing.")
        print(combined_df.isnull().sum())

    # # Add specific check for the target column
    # if combined_df['NET Claims Incurred'].isnull().any():
    #     print("Warning: NaNs detected in target column after preprocessing.")



    return combined_df


# def prepare_for_arima_ma(combined_df, target_column):
#     """
#     Prepare individual country datasets for ARIMA/MA modeling and save all preprocessed data in a single DataFrame.

#     Parameters:
#     - combined_df: The combined DataFrame containing data for all countries.
#     - target_column: The target variable for time series modeling.

#     Returns:
#     - A single DataFrame with all preprocessed data, including a 'Country' column.
#     """
#     if 'Date' not in combined_df.columns or target_column not in combined_df.columns:
#         raise ValueError("The combined_df must contain 'Date' and target_column for ARIMA/MA modeling.")
    
#     # Ensure the 'Date' column is datetime
#     combined_df['Date'] = pd.to_datetime(combined_df['Date'])

#     # Initialize an empty list to store processed data
#     processed_data = []

#     for country, group in combined_df.groupby('Country'):
#         # Extract necessary columns
#         arima_df = group[['Date', target_column]].dropna()

#         # Set Date as index
#         arima_df.set_index('Date', inplace=True)

#         # Infer and set frequency
#         inferred_freq = pd.infer_freq(arima_df.index)
#         if inferred_freq is None:
#             raise ValueError(f"Could not infer frequency for the time series data of country: {country}.")
#         arima_df.index.freq = inferred_freq

#         # Check for missing periods
#         missing_periods = pd.date_range(start=arima_df.index.min(), end=arima_df.index.max(), freq=inferred_freq).difference(arima_df.index)
#         if len(missing_periods) > 0:
#             print(f"Warning: Missing periods detected in the time series for country {country}: {missing_periods}")

#         # Add a 'Country' column and append to the processed data list
#         arima_df = arima_df.iloc[5:]  # Drop the first 5 rows
#         arima_df['Country'] = country
#         processed_data.append(arima_df)

#     # Concatenate all processed data into a single DataFrame
#     country_dfs = pd.concat(processed_data).reset_index()

#     return country_dfs

def prepare_for_arima_ma(combined_df, target_column):
    """
    Prepare individual country datasets for ARIMA/MA modeling.

    Parameters:
    - combined_df: The combined DataFrame containing data for all countries.
    - target_column: The target variable for time series modeling.

    Returns:
    - A dictionary with country names as keys and individual DataFrames as values.
    """
    #print("\n----- START Pre-processing prepare_for_arima_ma")
    if 'Date' not in combined_df.columns or target_column not in combined_df.columns:
        raise ValueError("The combined_df must contain 'Date' and target_column for ARIMA/MA modeling.")
    
    # Ensure the 'Date' column is datetime
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])

    # Split the combined DataFrame into individual DataFrames for each country
    country_dfs = {}
    for country, group in combined_df.groupby('Country'):
        # Extract necessary columns
        arima_df = group[['Date', target_column]].dropna() #.copy()
        #print(f"Initial arima_df shape: {arima_df.shape}")
        
        # Set Date as index
        arima_df.set_index('Date', inplace=True)
        #print(f"arima_df after setting 'Date' as index:\n{arima_df.head()}")
        
        # Infer and set frequency
        inferred_freq = pd.infer_freq(arima_df.index)
        #print(f"Inferred frequency: {inferred_freq}")
        if inferred_freq is None:
            raise ValueError(f"Could not infer frequency for the time series data of country: {country}.")
        arima_df.index.freq = inferred_freq
        
        # Check for missing periods
        missing_periods = pd.date_range(start=arima_df.index.min(), end=arima_df.index.max(), freq=inferred_freq).difference(arima_df.index)
        if len(missing_periods) > 0:
            print(f"Warning: Missing periods detected in the time series for country {country}: {missing_periods}")
        #print(f"---- ARIMA dataset size: {arima_df.shape[0]}") 
        # Store the processed DataFrame for the current country
        country_dfs[country] = arima_df.iloc[5:]
        #print(f"---- ARIMA dataset size: {country_dfs[country].shape[0]}") 
        #print("IS IT OK?", country, country_dfs[country])
        #print("inside", country_dfs)
    #print("\n----- END Pre-processing prepare_for_arima_ma")
    return country_dfs






def prepare_for_arima_ma_old(combined_df, target_column):
    
    # Extract only the Date and target columns
    if 'Date' not in combined_df.columns or target_column not in combined_df.columns:
        raise ValueError("The combined_df must contain 'Date' and target_column for ARIMA/MA modeling.")
    
    #print("\n----- Pre-processing ARIMA/MA dataset")

    # Extract necessary columns
    arima_df = combined_df[['Date', 'Country', target_column]].copy()
    #print(f"Initial arima_df shape: {arima_df.shape}")
    #print(f"Initial arima_df head:\n{arima_df.head()}")

    # Drop rows with NaN in the target column
    arima_df = arima_df.dropna(subset=[target_column])
    #print(f"Shape after dropping NaNs in target column: {arima_df.shape}")
    #print(f"Sample of arima_df after dropping NaNs:\n{arima_df.head()}")    

    # Set Date as index (very important for ARIMA)
    #arima_df.index = pd.PeriodIndex(arima_df.index, freq="Q") # or arima_df['Date'] = pd.to_datetime(arima_df['Date'])
    arima_df['Date'] = pd.to_datetime(arima_df['Date'])
    arima_df.set_index('Date', inplace=True)
    #print(f"arima_df after setting 'Date' as index:\n{arima_df.head()}")
    #print(f"arima_df index type: {type(arima_df.index)}")

    # Infer and validate frequency
    inferred_freq = pd.infer_freq(arima_df.index)
    #print(f"Inferred frequency: {inferred_freq}")
    if inferred_freq is None:
        raise ValueError("Could not infer a valid frequency for the time series data.")

    # After inferring the frequency. explicitly assign the inferred frequency to the index
    arima_df.index.freq = inferred_freq
    #print(f"arima_df index frequency after assignment: {arima_df.index.freq}")

    # Check for missing periods
    missing_periods = pd.date_range(start=arima_df.index.min(), end=arima_df.index.max(), freq=inferred_freq).difference(arima_df.index)
    if len(missing_periods) > 0:
        print(f"Missing periods in the time series: {missing_periods}")
    else:
        print("No missing periods in the time series.")

    # Final check
    #print(f"Final arima_df shape: {arima_df.shape}")
    #print(f"Final arima_df head:\n{arima_df.head()}")
    #print(f"arima_df index frequency: {arima_df.index.freq}")
    #print(f"---- ARIMA dataset size: {arima_df.shape[0]}")  #ARIMA dataset size: 31

    return arima_df.iloc[5:]  # Drop the first 5 rows of each country's data for ARIMA modeling





