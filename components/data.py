from io import StringIO
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
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
        return pd.Timestamp(year=year, month=quarter_mapping[quarter], day=1) #.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Conversion error with value: {quarter_str} -> {e}")
        return pd.NaT  # Return NaT for invalid formats

def load_and_process_uploaded_data(contents, filenames, existing_dataframes):
    # for content, filename in zip(contents, filenames):
    #     country_name = filename.split('.')[0]
    #     content_type, content_string = content.split(',')
    #     decoded = pd.read_csv(StringIO(content_string))

    #     if country_name in existing_dataframes:
    #         existing_dataframes[f"{country_name}_new"] = decoded
    #     else:
    #         existing_dataframes[country_name] = decoded

    for name, df in existing_dataframes.items():
        df.columns = df.columns.str.strip()
        print("give me value ",df.columns)
        
        for col in df.select_dtypes(include='object').columns:
            if 'date' not in col.lower(): # pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col].str.replace(' ', '').str.replace(',', ''), errors='coerce')
                except Exception:
                    continue

        # Explicitly check and apply the quarter-to-date conversion
        if 'Date' in df.columns.tolist():
            print(f"Before conversion, 'Date' sample values in {name}: {df['Date'].head()}")
            df['Date'] = df['Date'].apply(lambda x: convert_quarter_to_date(x) if isinstance(x, str) else x)
            print(f"After conversion, 'Date' sample values in {name}: {df['Date'].head()}")
        else:
            print(f"No 'Date' column found in {name}.")

        for col in df.select_dtypes(include='float64').columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())

    combined_df = pd.DataFrame()
    for country, df in existing_dataframes.items():
        df['Country'] = country
        combined_df = pd.concat([combined_df, df], axis=0) #, ignore_index=True

    le = LabelEncoder()
    combined_df['Country'] = le.fit_transform(combined_df['Country'])

    numeric_columns = combined_df.select_dtypes(include='float64').columns #.tolist()
    lagged_dataframes = [combined_df]  # list to store the main DF and lagged columns
    #print('numeric col', numeric_columns)
    #print('i want this', combined_df['NET Premiums Earned'].dtype)

    for col in numeric_columns:
        for lag in range(1, 9):
            combined_df[f'{col}_Lag{lag}'] = combined_df.groupby('Country')[col].shift(lag)
            # lagged_df = combined_df[numeric_columns].groupby('Country').shift(lag)
            # lagged_df = lagged_df.add_suffix(f"_Lag{lag}")
            # lagged_dataframes.append(lagged_df)


    # Extract Year and Quarter if 'Date' is properly formatted
    if 'Date' in combined_df.columns and pd.api.types.is_datetime64_any_dtype(combined_df['Date']):
        combined_df['Year'] = combined_df['Date'].dt.year
        combined_df['Quarter'] = combined_df['Date'].dt.quarter
    else:
        combined_df['Year'] = np.nan
        combined_df['Quarter'] = np.nan

    combined_df = combined_df.groupby('Country').apply(lambda group: group.ffill().bfill()).reset_index(drop=True)

    return combined_df












# Export the variables
#__all__ = ['combined_df']