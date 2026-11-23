"""Prepare supplied L9 visits and attach polygon-based amenity exposure."""

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from pyproj import Transformer

from k_visitation import DEFAULT_AMENITY_COLUMNS


def aggregate_visits(visits, keys, *, sort=False):
    """Sum visit frequencies for each group of records."""
    return visits.groupby(keys, sort=sort, as_index=False)['visit_freq'].sum()


def prepare_locations(visits, pois):
    """Aggregate complete L9 records, calculate centre distances and attach POIs."""
    keys = ['user_id', 'stay_gid9', 'home_gid9', 'work_gid9']
    cells = pd.unique(visits[['stay_gid9', 'home_gid9']].to_numpy().ravel())
    result = aggregate_visits(visits, keys)
    centres = pd.DataFrame([h3.cell_to_latlng(cell) for cell in cells],
                           index=cells, columns=['lat', 'lon'])
    home = centres.loc[result.home_gid9].to_numpy()
    destination = centres.loc[result.stay_gid9].to_numpy()
    result['home_dist'] = centre_distance(home, destination)
    return attach_poi(result, pois)


def centre_distance(home, destination):
    """Return Euclidean distances in metres between L9 centres in EPSG:3067."""
    project = Transformer.from_crs(4326, 3067, always_xy=True)
    home_xy = np.column_stack(project.transform(home[:, 1], home[:, 0]))
    destination_xy = np.column_stack(project.transform(destination[:, 1], destination[:, 0]))
    return np.linalg.norm(destination_xy - home_xy, axis=1)


def attach_poi(visits, pois):
    """Count the ten amenity classes intersecting each visited L9 polygon."""
    categories = list(DEFAULT_AMENITY_COLUMNS)
    cells = visits.stay_gid9.unique()
    polygons = [Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(cell)]) for cell in cells]
    grid = gpd.GeoDataFrame({'stay_gid9': cells}, geometry=polygons, crs=4326).to_crs(3067)
    points = pois[['category', 'geometry']].dropna().to_crs(3067)
    matched = gpd.sjoin(points.loc[points.category.isin(categories)], grid,
                        how='inner', predicate='intersects')
    counts = matched.groupby(['stay_gid9', 'category']).size().unstack(fill_value=0)
    counts = counts.reindex(columns=categories, fill_value=0)
    result = visits.join(counts, on='stay_gid9')
    result[categories] = result[categories].fillna(0).astype('int32')
    return result
