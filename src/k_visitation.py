import pandas as pd
import numpy as np

def calculate_k_places(places_df, amenity_list, smallest_values, 
                                sort_column='home_dist', ascending=True, 
                                k_type='k_dist'):
    """
    Calculate K-visitation places for users.
    
    Scenarios:
    1. Complete: Requirements satisfied - mark cumulative places as 1, stop
    2. Incomplete: Requirements not satisfied - mark ALL places as 1
    
    Parameters:
    -----------
    places_df : DataFrame
        Places dataframe with user_id and amenity columns
    amenity_list : list
        List of amenity column names
    smallest_values : Series or array
        Minimum required values for each amenity
    sort_column : str
        Column to sort by ('home_dist' for K-dist, 'visit_freq' for K-freq)
    ascending : bool
        Sort order (True for distance, False for frequency)
    k_type : str
        Type identifier for output column
    
    Returns:
    --------
    DataFrame : Original dataframe with K-place indicators added
    """
    
    # Sort data
    places_sorted = places_df.sort_values(
        by=['user_id', sort_column], 
        ascending=[True, ascending]
    ).reset_index(drop=True)
    
    # Fill missing amenity values
    places_sorted[amenity_list] = places_sorted[amenity_list].fillna(0)
    
    # Convert to numpy for faster computation
    smallest_values_np = smallest_values.to_numpy() if hasattr(smallest_values, 'to_numpy') else np.array(smallest_values)
    
    # Initialize result arrays
    k_indicator = np.zeros(len(places_sorted), dtype=np.int8)
    k_status = np.full(len(places_sorted), 'unassigned', dtype=object)
    
    # Group by user for processing
    user_groups = places_sorted.groupby('user_id')
    
    for user_id, user_data in user_groups:
        indices = user_data.index.tolist()
        user_poi = user_data[amenity_list].to_numpy()
        
        # Initialize tracking variables
        total_poi_access = np.zeros_like(smallest_values_np)
        k_user = np.zeros(len(indices), dtype=np.int8)
        requirements_met = False
        
        # Process each place for this user to find completion point
        for idx, row_poi in enumerate(user_poi):
            # Add current place's amenities
            total_poi_access += row_poi
            
            # Check if requirements are met after adding this place
            is_complete = np.all(total_poi_access >= smallest_values_np)
            
            if is_complete:
                # SCENARIO 1: COMPLETE - Mark cumulative places (0 to idx) as K-places
                k_user[:idx+1] = 1
                requirements_met = True
                break
        
        # SCENARIO 2: INCOMPLETE - If requirements not met after all places
        if not requirements_met:
            # Mark ALL places as K-places
            k_user[:] = 1
        
        # Assign results back to main arrays
        for i, idx in enumerate(indices):
            k_indicator[idx] = k_user[i]
        
        # Determine completion status
        if requirements_met:
            status = 'complete'    # Requirements fully met
        else:
            status = 'incomplete'  # Requirements not met, all places selected
        
        # Apply status to all places for this user
        for idx in indices:
            k_status[idx] = status
    
    # Add results to dataframe
    places_sorted[f'{k_type}'] = k_indicator
    places_sorted[f'{k_type}_status'] = k_status
    
    return places_sorted

# Wrapper function for calculating both K-dist and K-freq
def calculate_both_k_places(places_df, amenity_list, smallest_values):
    """Calculate both K-dist and K-freq places with corrected logic"""
    
    # Calculate K-dist places (sorted by distance, ascending)
    places_with_kdist = calculate_k_places(
        places_df=places_df,
        amenity_list=amenity_list,
        smallest_values=smallest_values,
        sort_column='home_dist',
        ascending=True,
        k_type='k_dist',
    )
    
    # Calculate K-freq places (sorted by frequency, descending)
    places_with_both = calculate_k_places(
        places_df=places_with_kdist,
        amenity_list=amenity_list,
        smallest_values=smallest_values,
        sort_column='visit_freq',
        ascending=False,
        k_type='k_freq',
    )
    
    return places_with_both

def calculate_qk_alignment(places_df, user_col='user_id', k_freq_col='k_freq', k_dist_col='k_dist'):
    """
    Calculate qK alignment using Jaccard similarity index for each user
    
    This function:
    1. Classifies each place into categories based on K-freq and K-dist indicators
    2. Calculates Jaccard similarity index for each user
    3. Returns user-level alignment metrics and place categorizations
    
    Parameters:
    -----------
    places_df : DataFrame
        Stay locations dataframe with user_id and K-place indicators
    user_col : str
        Column name for user identifier
    k_freq_col : str
        Column name for K-freq indicator (0 or 1)
    k_dist_col : str
        Column name for K-dist indicator (0 or 1)
    
    Returns:
    --------
    tuple : (user_alignment_df, places_with_categories_df)
        - user_alignment_df: User-level qK metrics
        - places_with_categories_df: Original dataframe with place categories added
    """
    
    # Create a copy to avoid modifying original data
    places_analysis = places_df.copy()
    
    # Ensure K-place indicators are binary (0 or 1)
    places_analysis[k_freq_col] = places_analysis[k_freq_col].fillna(0).astype(int)
    places_analysis[k_dist_col] = places_analysis[k_dist_col].fillna(0).astype(int)
    
    # Step 1: Assign place categories based on K-freq and K-dist values
    def get_place_category(row):
        k_freq = row[k_freq_col]
        k_dist = row[k_dist_col]
        
        if k_freq == 1 and k_dist == 1:
            return 'f1d1'  # Both methods identify this place
        elif k_freq == 1 and k_dist == 0:
            return 'f1d0'  # Only K-freq identifies this place
        elif k_freq == 0 and k_dist == 1:
            return 'f0d1'  # Only K-dist identifies this place
        elif k_freq == 0 and k_dist == 0:
            return 'f0d0'  # Neither method identifies this place
        else:
            return 'other'
    
    places_analysis['k_type'] = places_analysis.apply(get_place_category, axis=1)
    
    # Step 2: Aggregate by user to count places in each category
    user_place_counts = places_analysis.groupby(user_col)['k_type'].value_counts().unstack(fill_value=0)

    # Ensure all categories exist in the dataframe
    for category in ['f1d1', 'f1d0', 'f0d1', 'f0d0']:
        if category not in user_place_counts.columns:
            user_place_counts[category] = 0
    
    user_place_counts = user_place_counts.reset_index()
    
    # Step 3: Calculate Jaccard similarity for each user
    def calculate_jaccard(row):
        """Jaccard = |A ∩ B| / |A ∪ B| = f1d1 / (f1d1 + f1d0 + f0d1)"""
        numerator = row['f1d1']
        denominator = row['f1d1'] + row['f1d0'] + row['f0d1']
        return numerator / denominator if denominator > 0 else 0.0
    
    # Calculate alignment metrics
    user_place_counts['qk'] = user_place_counts.apply(calculate_jaccard, axis=1)
    
    return user_place_counts
