import pandas as pd
import numpy as np
import os
from io import StringIO


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


#---------------- Data Preparation --------------------------------------------------------------





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