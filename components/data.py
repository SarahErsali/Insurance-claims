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



def prepare_for_arima_ma(combined_df, target_column):
    
    # Extract only the Date and target columns
    if 'Date' not in combined_df.columns or target_column not in combined_df.columns:
        raise ValueError("The combined_df must contain 'Date' and target_column for ARIMA/MA modeling.")
    
    #print("\n----- Pre-processing ARIMA/MA dataset")

    # Extract necessary columns
    arima_df = combined_df[['Date', target_column]].copy()
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





