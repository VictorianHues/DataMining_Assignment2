import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import scipy.sparse as sp


def load_feature_data(path):
    """Load feature data more efficiently"""
    df = pd.read_csv(path)
    srch_ids = df['srch_id'].values
    X = df.drop(columns=['srch_id']).fillna(0).values
    return X, srch_ids


def fit_knn_model(X, k):
    """Fit KNN model with better algorithm for high dimensions"""
    # Use more efficient algorithm for high dimensions
    knn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', n_jobs=-1)
    knn.fit(X)
    return knn


def pre_aggregate_batch(full_data_path, batch_size=1000000):
    """Pre-aggregate data more efficiently using batching and vectorized operations"""
    print("Pre-aggregating full dataset...")
    
    # Initialize dataframes to store aggregated data
    srch_prop_agg = []
    prop_agg = []
    
    # Process in larger chunks for better I/O performance
    for chunk in tqdm(pd.read_csv(full_data_path, chunksize=batch_size), desc="Reading chunks"):
        # Group by search-property pairs
        sp_group = chunk.groupby(['srch_id', 'prop_id']).agg({
            'click_bool': 'sum',
            'booking_bool': 'sum'
        }).reset_index()
        
        # Add count column
        sp_group['count'] = 1
        srch_prop_agg.append(sp_group)
        
        # Group by property
        p_group = chunk.groupby('prop_id').agg({
            'click_bool': 'sum',
            'booking_bool': 'sum'
        }).reset_index()
        
        p_group['count'] = chunk.groupby('prop_id').size().values
        prop_agg.append(p_group)
    
    # Combine all batches
    srch_prop_stats = pd.concat(srch_prop_agg).groupby(['srch_id', 'prop_id']).agg({
        'click_bool': 'sum',
        'booking_bool': 'sum',
        'count': 'sum'
    }).reset_index()
    
    prop_stats = pd.concat(prop_agg).groupby('prop_id').agg({
        'click_bool': 'sum',
        'booking_bool': 'sum',
        'count': 'sum'
    }).reset_index()
    
    # Rename columns for clarity
    srch_prop_stats.columns = ['srch_id', 'prop_id', 'clicks', 'books', 'count']
    prop_stats.columns = ['prop_id', 'clicks', 'books', 'count']
    
    return srch_prop_stats, prop_stats


def batch_get_neighbors(knn, X, srch_ids, batch_size=1000, k=50):
    """Get neighbors in batches rather than one at a time"""
    all_neighbors = []
    n_samples = len(srch_ids)
    
    for start_idx in tqdm(range(0, n_samples, batch_size), desc="Finding neighbors in batches"):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_X = X[start_idx:end_idx]
        
        # Get k+1 neighbors (including self)
        distances, indices = knn.kneighbors(batch_X, n_neighbors=k+1)
        
        # Remove self (first column)
        neighbor_indices = indices[:, 1:]
        
        # Map indices to srch_ids
        batch_neighbors = []
        for i, user_idx in enumerate(range(start_idx, end_idx)):
            target_id = srch_ids[user_idx]
            neighbor_ids = srch_ids[neighbor_indices[i]]
            batch_neighbors.append((target_id, neighbor_ids))
            
        all_neighbors.extend(batch_neighbors)
    
    return all_neighbors


def create_user_property_matrix(srch_prop_stats):
    """Create a sparse matrix of user-property interactions for efficient lookups"""
    # Create mapping dictionaries for search_id and prop_id to matrix indices
    unique_searches = srch_prop_stats['srch_id'].unique()
    unique_props = srch_prop_stats['prop_id'].unique()
    
    srch_to_idx = {srch_id: idx for idx, srch_id in enumerate(unique_searches)}
    prop_to_idx = {prop_id: idx for idx, prop_id in enumerate(unique_props)}
    idx_to_prop = {idx: prop_id for prop_id, idx in prop_to_idx.items()}
    
    # Create sparse matrices for interactions
    n_searches = len(unique_searches)
    n_props = len(unique_props)
    
    # Map srch_ids and prop_ids to their matrix indices
    row_indices = [srch_to_idx[srch_id] for srch_id in srch_prop_stats['srch_id']]
    col_indices = [prop_to_idx[prop_id] for prop_id in srch_prop_stats['prop_id']]
    
    # Create sparse matrices
    interaction_matrix = sp.csr_matrix(
        (np.ones(len(row_indices)), (row_indices, col_indices)), 
        shape=(n_searches, n_props)
    )
    
    click_matrix = sp.csr_matrix(
        (srch_prop_stats['clicks'].values, (row_indices, col_indices)), 
        shape=(n_searches, n_props)
    )
    
    book_matrix = sp.csr_matrix(
        (srch_prop_stats['books'].values, (row_indices, col_indices)), 
        shape=(n_searches, n_props)
    )
    
    count_matrix = sp.csr_matrix(
        (srch_prop_stats['count'].values, (row_indices, col_indices)), 
        shape=(n_searches, n_props)
    )
    
    return {
        'interaction': interaction_matrix,
        'clicks': click_matrix,
        'books': book_matrix,
        'count': count_matrix,
        'srch_to_idx': srch_to_idx,
        'prop_to_idx': prop_to_idx,
        'idx_to_prop': idx_to_prop,
        'unique_searches': unique_searches
    }


def calculate_property_scores(matrices, prop_stats, click_weight, book_weight):
    """Calculate property scores using vectorized operations"""
    # Compute combined score for all properties
    prop_df = prop_stats.copy()
    prop_df['raw_score'] = (click_weight * prop_df['clicks'] + book_weight * prop_df['books']) / np.maximum(1, prop_df['count'])
    prop_df['smoothing'] = np.log1p(prop_df['count'])
    prop_df['combined_score'] = prop_df['raw_score'] * prop_df['smoothing']
    
    return prop_df[['prop_id', 'combined_score']]


def generate_recommendations_batch(
    neighbor_data, 
    matrices, 
    prop_scores,
    top_n=10,
    batch_size=100
):
    """Generate recommendations in batches using vectorized operations"""
    all_recommendations = []
    
    # Process users in batches
    for batch_start in tqdm(range(0, len(neighbor_data), batch_size), desc="Generating recommendations"):
        batch_end = min(batch_start + batch_size, len(neighbor_data))
        batch_neighbors = neighbor_data[batch_start:batch_end]
        
        batch_recs = []
        
        for target_srch_id, neighbor_srch_ids in batch_neighbors:
            # Get matrix indices
            if target_srch_id not in matrices['srch_to_idx']:
                continue  # Skip if search ID not found
                
            target_idx = matrices['srch_to_idx'][target_srch_id]
            
            # Get properties seen by target user
            target_row = matrices['interaction'][target_idx].toarray().flatten()
            seen_prop_indices = np.where(target_row > 0)[0]
            seen_prop_ids = {matrices['idx_to_prop'][idx] for idx in seen_prop_indices}
            
            # Filter properties not seen by target
            candidate_props = prop_scores[~prop_scores['prop_id'].isin(seen_prop_ids)]
            
            # Get top N recommendations
            recommendations = candidate_props.nlargest(top_n, 'combined_score')
            
            # Add to results
            for _, rec in recommendations.iterrows():
                batch_recs.append({
                    'srch_id': target_srch_id,
                    'prop_id': rec['prop_id']
                })
        
        all_recommendations.extend(batch_recs)
    
    return pd.DataFrame(all_recommendations)


def recommend_for_all_users(
    agg_path,
    full_data_path,
    output_path,
    k_neighbors=50,
    top_n=10,
    click_weight=1.0,
    book_weight=4.0,
    batch_size=1000
):
    """Generate recommendations for all users with optimized performance"""
    print("Loading feature data...")
    X, srch_ids = load_feature_data(agg_path)

    print("Fitting KNN model...")
    knn = fit_knn_model(X, k_neighbors)
    
    print("Pre-aggregating data...")
    srch_prop_stats, prop_stats = pre_aggregate_batch(full_data_path)
    
    print("Creating user-property matrices...")
    matrices = create_user_property_matrix(srch_prop_stats)
    
    print("Finding neighbors in batches...")
    neighbor_data = batch_get_neighbors(knn, X, srch_ids, batch_size=batch_size, k=k_neighbors)
    
    print("Calculating property scores...")
    prop_scores = calculate_property_scores(matrices, prop_stats, click_weight, book_weight)
    
    print("Generating recommendations...")
    recommendations = generate_recommendations_batch(
        neighbor_data, 
        matrices, 
        prop_scores,
        top_n=top_n,
        batch_size=batch_size
    )
    
    recommendations.to_csv(output_path, index=False)
    print(f"✅ Recommendations saved to: {output_path}")


# Example usage
