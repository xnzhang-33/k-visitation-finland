import pandas as pd
import numpy as np


DEFAULT_AMENITY_COLUMNS = [
    'CIVIC_RELIGION',
    'CULTURE',
    'DINING',
    'EDUCATION',
    'FITNESS',
    'GROCERIES',
    'HEALTHCARE',
    'RETAIL',
    'SERVICE',
    'TRANSPORT',
]

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

def calculate_k_places_v2(places_df, amenity_list,
                          sort_column='home_dist', ascending=True,
                          k_type='k_dist'):
    """
    K-visitation v2 with per-user dynamic thresholds and POI-richness tie-break.

    Differences vs `calculate_k_places`:
    - Per-user dynamic threshold: only amenities the user actually visits are
      required (>=1 of each). Users not covering all 10 categories are not
      penalised.
    - Tie-break: within ties on `sort_column`, stays offering more POI
      categories come first (deterministic selection).
    - No internal filtering: caller drops home rows / chooses to keep
      `visit_freq == 1` rows beforehand.

    Returns the input dataframe (sorted) with `{k_type}` and
    `{k_type}_status` columns added.
    """
    places = places_df.copy()
    places[amenity_list] = places[amenity_list].fillna(0)

    # Tie-break key: number of distinct amenity categories at the stay
    places['_n_amenities'] = (places[amenity_list] > 0).sum(axis=1)

    places_sorted = places.sort_values(
        by=['user_id', sort_column, '_n_amenities'],
        ascending=[True, ascending, False]
    ).reset_index(drop=True)

    # Per-user mask of amenities the user actually visits (>=1 anywhere)
    user_amenity_mask = (
        places_sorted.groupby('user_id')[amenity_list].sum() > 0
    )

    k_indicator = np.zeros(len(places_sorted), dtype=np.int8)
    k_status = np.full(len(places_sorted), 'unassigned', dtype=object)

    for user_id, user_data in places_sorted.groupby('user_id'):
        indices = user_data.index.to_numpy()
        user_poi = user_data[amenity_list].to_numpy()

        mask = user_amenity_mask.loc[user_id].to_numpy()
        # If the user has no amenity coverage at all, mark every stay as K
        if not mask.any():
            k_indicator[indices] = 1
            k_status[indices] = 'no_amenity'
            continue

        total = np.zeros(len(amenity_list))
        k_user = np.zeros(len(indices), dtype=np.int8)
        requirements_met = False

        for idx, row_poi in enumerate(user_poi):
            total += row_poi
            if np.all(total[mask] >= 1):
                k_user[:idx + 1] = 1
                requirements_met = True
                break

        if not requirements_met:
            k_user[:] = 1

        k_indicator[indices] = k_user
        k_status[indices] = 'complete' if requirements_met else 'incomplete'

    places_sorted[f'{k_type}'] = k_indicator
    places_sorted[f'{k_type}_status'] = k_status
    places_sorted = places_sorted.drop(columns=['_n_amenities'])

    return places_sorted


def calculate_both_k_places_v2(places_df, amenity_list):
    """Compute both K-dist and K-freq using the v2 rules.

    Caller is responsible for:
    - Dropping `stay_gid9 == home_gid9` rows beforehand.
    - Including `visit_freq == 1` rows if desired.
    """
    with_kdist = calculate_k_places_v2(
        places_df=places_df,
        amenity_list=amenity_list,
        sort_column='home_dist',
        ascending=True,
        k_type='k_dist',
    )
    with_both = calculate_k_places_v2(
        places_df=with_kdist,
        amenity_list=amenity_list,
        sort_column='visit_freq',
        ascending=False,
        k_type='k_freq',
    )
    return with_both


def calculate_qk_weighted(places_df, user_col='user_id',
                          k_freq_col='k_freq', k_dist_col='k_dist',
                          weight_col='visit_freq'):
    """
    qK alignment with frequency weighting in addition to unweighted Jaccard.

    Unweighted (same as `calculate_qk_alignment`):
        qk = |f1d1| / |f1d1 ∪ f1d0 ∪ f0d1|

    Weighted:
        qk_weighted = sum(w | f1d1) / sum(w | f1d1 ∪ f1d0 ∪ f0d1)

    A high-`visit_freq` place sitting in `f1d0` (in K-freq, not K-dist)
    inflates the denominator and pulls `qk_weighted` below `qk`.

    Returns a DataFrame keyed by `user_col` with both metrics and category
    counts/weights.
    """
    df = places_df.copy()
    df[k_freq_col] = df[k_freq_col].fillna(0).astype(int)
    df[k_dist_col] = df[k_dist_col].fillna(0).astype(int)
    df[weight_col] = df[weight_col].fillna(0)

    cat = np.where(
        (df[k_freq_col] == 1) & (df[k_dist_col] == 1), 'f1d1',
        np.where(
            (df[k_freq_col] == 1) & (df[k_dist_col] == 0), 'f1d0',
            np.where(
                (df[k_freq_col] == 0) & (df[k_dist_col] == 1), 'f0d1',
                'f0d0'
            )
        )
    )
    df['k_type'] = cat

    counts = (
        df.groupby(user_col)['k_type'].value_counts().unstack(fill_value=0)
    )
    weights = (
        df.groupby([user_col, 'k_type'])[weight_col].sum().unstack(fill_value=0)
    )
    for c in ['f1d1', 'f1d0', 'f0d1', 'f0d0']:
        if c not in counts.columns:
            counts[c] = 0
        if c not in weights.columns:
            weights[c] = 0.0

    counts = counts.rename(columns={c: f'{c}' for c in counts.columns})
    weights = weights.rename(columns={c: f'{c}_w' for c in weights.columns})

    out = counts.join(weights).reset_index()

    union_n = out['f1d1'] + out['f1d0'] + out['f0d1']
    union_w = out['f1d1_w'] + out['f1d0_w'] + out['f0d1_w']

    out['qk'] = np.where(union_n > 0, out['f1d1'] / union_n, 0.0)
    out['qk_weighted'] = np.where(union_w > 0, out['f1d1_w'] / union_w, 0.0)

    return out


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


def calculate_k_visitation(
    places_df,
    amenity_columns=DEFAULT_AMENITY_COLUMNS,
    *,
    user_col='user_id',
    location_col=None,
    frequency_col='visit_freq',
    distance_col='home_dist',
):
    """Calculate ``K_freq``, ``K_dist`` and frequency-weighted ``q_K``.

    Each input row must represent one non-home visited location. For each user,
    the coverage target is the set of amenity categories observed anywhere in
    that user's portfolio. Locations are added sequentially by decreasing visit
    frequency for ``K_freq`` and increasing home distance for ``K_dist`` until
    that same target is covered.

    Ties are resolved by ``location_col`` when supplied, otherwise by input row
    order. The returned place table retains input order and adds ``k_freq``,
    ``k_dist`` and ``k_type``. The summary reports the manuscript's
    visit-frequency-weighted ``qk`` as well as the unweighted overlap for audit.
    """
    amenity_columns = list(amenity_columns)
    required = [user_col, frequency_col, distance_col, *amenity_columns]
    if location_col is not None:
        required.append(location_col)

    missing = [column for column in required if column not in places_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not amenity_columns:
        raise ValueError("At least one amenity column is required")
    if places_df.empty:
        raise ValueError("The visitation portfolio is empty")
    if places_df[required].isna().any().any():
        raise ValueError("Required K-visitation fields must not contain missing values")
    if (places_df[frequency_col] < 0).any() or (places_df[distance_col] < 0).any():
        raise ValueError("Visit frequencies and home distances must be non-negative")
    if (places_df[amenity_columns] < 0).any().any():
        raise ValueError("Amenity counts must be non-negative")
    if location_col is not None and places_df.duplicated([user_col, location_col]).any():
        raise ValueError("Each user-location pair must appear exactly once")

    places = places_df.copy().reset_index(drop=True)
    places['_input_order'] = np.arange(len(places))
    places['k_freq'] = np.int8(0)
    places['k_dist'] = np.int8(0)

    tie_column = location_col or '_input_order'

    for _, user_places in places.groupby(user_col, sort=False):
        target = (user_places[amenity_columns] > 0).any(axis=0).to_numpy()
        if not target.any():
            continue

        orderings = {
            'k_freq': ([frequency_col, tie_column], [False, True]),
            'k_dist': ([distance_col, tie_column], [True, True]),
        }
        for indicator, (sort_columns, ascending) in orderings.items():
            ordered = user_places.sort_values(
                sort_columns,
                ascending=ascending,
                kind='mergesort',
            )
            cumulative_coverage = (
                (ordered[amenity_columns] > 0)
                .cummax()
                .to_numpy()
            )
            completed = np.flatnonzero(cumulative_coverage[:, target].all(axis=1))
            if completed.size:
                selected = ordered.index[: completed[0] + 1]
                places.loc[selected, indicator] = np.int8(1)

    places['k_type'] = np.select(
        [
            (places['k_freq'] == 1) & (places['k_dist'] == 1),
            (places['k_freq'] == 1) & (places['k_dist'] == 0),
            (places['k_freq'] == 0) & (places['k_dist'] == 1),
        ],
        ['f1d1', 'f1d0', 'f0d1'],
        default='f0d0',
    )

    in_intersection = (places['k_freq'] == 1) & (places['k_dist'] == 1)
    in_union = (places['k_freq'] == 1) | (places['k_dist'] == 1)
    audit = places.assign(
        _intersection_n=in_intersection.astype(int),
        _union_n=in_union.astype(int),
        _intersection_visits=places[frequency_col].where(in_intersection, 0),
        _union_visits=places[frequency_col].where(in_union, 0),
    )
    summary = (
        audit.groupby(user_col, sort=False)
        .agg(
            k_freq_size=('k_freq', 'sum'),
            k_dist_size=('k_dist', 'sum'),
            intersection_size=('_intersection_n', 'sum'),
            union_size=('_union_n', 'sum'),
            intersection_visits=('_intersection_visits', 'sum'),
            union_visits=('_union_visits', 'sum'),
        )
        .reset_index()
    )
    summary['qk'] = np.divide(
        summary['intersection_visits'],
        summary['union_visits'],
        out=np.zeros(len(summary), dtype=float),
        where=summary['union_visits'].to_numpy() > 0,
    )
    summary['qk_unweighted'] = np.divide(
        summary['intersection_size'],
        summary['union_size'],
        out=np.zeros(len(summary), dtype=float),
        where=summary['union_size'].to_numpy() > 0,
    )

    return places.drop(columns=['_input_order']), summary
