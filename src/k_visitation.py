"""Ordered amenity coverage and visit-frequency-weighted overlap."""

import numpy as np
import pandas as pd

DEFAULT_AMENITY_COLUMNS = [
    'CIVIC_RELIGION', 'CULTURE', 'DINING', 'EDUCATION', 'FITNESS',
    'GROCERIES', 'HEALTHCARE', 'RETAIL', 'SERVICE', 'TRANSPORT',
]


def select_prefix(places, categories, user_col, order, ascending):
    """Select each user's coverage prefix, retaining input order for equal ranks."""
    selected = pd.Series(0, index=places.index, dtype='int8')
    for _, group in places.groupby(user_col, sort=False):
        ordered = group.sort_values(order, ascending=ascending, kind='mergesort')
        amenities = ordered[categories].to_numpy() > 0
        target = amenities.any(axis=0)
        coverage = np.vstack([np.zeros((1, len(categories)), dtype=bool),
                              np.maximum.accumulate(amenities, axis=0)])
        stop = np.flatnonzero(coverage[:, target].all(axis=1))[0]
        selected.loc[ordered.index[:stop]] = 1
    return selected


def calculate_overlap(places, user_col, frequency_col):
    """Summarise visit-frequency-weighted overlap."""
    shared = places.k_freq.eq(1) & places.k_dist.eq(1)
    union = places.k_freq.eq(1) | places.k_dist.eq(1)
    audit = places.assign(
        _shared_visits=places[frequency_col].where(shared, 0),
        _union_visits=places[frequency_col].where(union, 0),
    )
    result = audit.groupby(user_col, sort=False).agg(
        k_freq_size=('k_freq', 'sum'), k_dist_size=('k_dist', 'sum'),
        intersection_visits=('_shared_visits', 'sum'), union_visits=('_union_visits', 'sum'),
    ).reset_index()
    result['qk_weighted'] = result.intersection_visits.div(result.union_visits.replace(0, np.nan)).fillna(0)
    return result


def calculate_k_visitation(
    places_df, amenity_columns=DEFAULT_AMENITY_COLUMNS, *,
    user_col='user_id', frequency_col='visit_freq', distance_col='home_dist',
):
    """Construct K sets independently by distance and frequency, then calculate overlap."""
    categories = list(amenity_columns)
    places = places_df.copy().reset_index(drop=True)
    places['k_dist'] = select_prefix(places, categories, user_col, distance_col, True)
    places['k_freq'] = select_prefix(places, categories, user_col, frequency_col, False)
    places['k_type'] = np.select(
        [places.k_freq.eq(1) & places.k_dist.eq(1), places.k_freq.eq(1), places.k_dist.eq(1)],
        ['f1d1', 'f1d0', 'f0d1'], default='f0d0',
    )
    return places, calculate_overlap(places, user_col, frequency_col)
