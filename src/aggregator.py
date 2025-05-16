import pandas as pd

def process_chunk(chunk, group_key='srch_id', mean_columns=None, first_columns=None):
    """
    Aggregates a single chunk of data by computing the mean for some columns and
    the first value for others, grouped by a key.
    """
    if mean_columns is None:
        mean_columns = []
    if first_columns is None:
        first_columns = []

    agg_dict = {col: 'mean' for col in mean_columns}
    agg_dict.update({col: 'first' for col in first_columns})

    agg_chunk = chunk.groupby(group_key).agg(agg_dict)

    # Count of items per group
    count_series = chunk.groupby(group_key).size().rename('hotel_count')
    agg_chunk = agg_chunk.join(count_series)

    return agg_chunk

def finalize_aggregation(aggregated_chunks, group_key='srch_id', mean_columns=None, first_columns=None):
    """
    Combines and re-aggregates a list of chunks, if needed.
    """
    if mean_columns is None:
        mean_columns = []
    if first_columns is None:
        first_columns = []

    if len(aggregated_chunks) > 1:
        combined = pd.concat(aggregated_chunks)
        agg_dict = {col: 'mean' for col in mean_columns}
        agg_dict.update({col: 'first' for col in first_columns})
        agg_dict['hotel_count'] = 'sum'
        final_df = combined.groupby(group_key).agg(agg_dict)
    else:
        final_df = aggregated_chunks[0]

    return final_df.reset_index()
