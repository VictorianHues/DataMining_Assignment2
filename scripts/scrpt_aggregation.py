import pandas as pd
from src.aggregator import *

# Configuration
chunk_size = 500_000
input_file = 'data/test_set_VU_DM_engineered.csv'
output_file = 'data/aggregated_data.csv'
group_key = 'srch_id'

# Columns to keep
columns = [
    'srch_id',
    'prop_starrating',
    'prop_review_score',
    'prop_brand_bool',
    'prop_location_score1',
    'price_usd',
    'promotion_flag',
    'srch_length_of_stay',
    'srch_booking_window',
    'srch_adults_count',
    'srch_children_count',
    'srch_room_count'
]

# Dtypes
dtypes = {
    'srch_id': 'int32',
    'prop_starrating': 'float32',
    'prop_review_score': 'float32',
    'prop_brand_bool': 'int8',
    'prop_location_score1': 'float32',
    'price_usd': 'float32',
    'promotion_flag': 'int8',
    'srch_length_of_stay': 'int16',
    'srch_booking_window': 'int16',
    'srch_adults_count': 'int8',
    'srch_children_count': 'int8',
    'srch_room_count': 'int8'
}

# Aggregation configs
mean_columns = [
    'prop_starrating',
    'prop_review_score',
    'prop_brand_bool',
    'prop_location_score1',
    'price_usd',
    'promotion_flag',
    'srch_length_of_stay',
    'srch_booking_window',
    'srch_adults_count',
    'srch_children_count',
    'srch_room_count'
]



# Aggregation process
aggregated_chunks = []
for chunk in pd.read_csv(input_file, usecols=columns, dtype=dtypes, chunksize=chunk_size):
    processed = process_chunk(chunk, group_key, mean_columns)
    aggregated_chunks.append(processed)

# Finalize and save
final_df = finalize_aggregation(aggregated_chunks, group_key, mean_columns)
final_df.to_csv(output_file, index=False)
print(f"Successfully aggregated data to {final_df.shape[0]} rows")