import pandas as pd

def read_csv(file_path):
    try:
        data_frame = pd.read_csv(file_path)
        print(f"CSV file read successfully: {file_path}")

        data_frame.replace("NULL", pd.NA, inplace=True) # Replace "NULL" with NaN

        if 'date_time' in data_frame.columns:
            data_frame['date_time'] = pd.to_datetime(data_frame['date_time']) # Convert 'date_time' to datetime format
            numeric_cols = data_frame.columns.drop(['date_time']) # Exclude 'date_time' from numeric conversion
        else:
            numeric_cols = data_frame.columns
        data_frame[numeric_cols] = data_frame[numeric_cols].apply(pd.to_numeric, errors='coerce') # Convert other columns to numeric, coercing errors to NaN

        return data_frame
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
def dataframe_to_csv(data_frame, file_path):
    try:
        data_frame.to_csv(file_path, index=False)
        print(f"DataFrame saved to {file_path}")
    except Exception as e:
        print(f"Error saving DataFrame to CSV: {e}")