"""Summarise recurrent and nearest-option distances by amenity class."""

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Polygon


# Source POI class mapped to its output identifier and broad category.
AMENITY_CLASSES = {
    "Airport": ("airport", "Transport"),
    "Amusement": ("amusement", "Culture"),
    "Appliances Store": ("appliances_store", "Retail"),
    "Auto Service": ("auto_service", "Service"),
    "Auto/Home Supply": ("auto_home_supply", "Retail"),
    "Bakery": ("bakery", "Groceries"),
    "Barber": ("barber", "Service"),
    "Beach": ("beach", "Park"),
    "Beauty Salon": ("beauty_salon", "Service"),
    "Book Store": ("book_store", "Retail"),
    "Bus Stop": ("bus_stop", "Transport"),
    "Cafe": ("cafe", "Dining"),
    "Car Dealer": ("car_dealer", "Retail"),
    "Car Rental": ("car_rental", "Retail"),
    "Car Wash": ("car_wash", "Service"),
    "Casino": ("casino", "Retail"),
    "Cinema": ("cinema", "Culture"),
    "Clinic": ("clinic", "Healthcare"),
    "Clothing & Accessories": ("clothing___accessories", "Retail"),
    "Concert Hall": ("concert_hall", "Culture"),
    "Convention Center": ("convention_center", "Civic & Religion"),
    "Cultural Center": ("cultural_center", "Culture"),
    "Daycare": ("daycare", "Education"),
    "Dentist": ("dentist", "Healthcare"),
    "Electrical Repair": ("electrical_repair", "Service"),
    "Employment Services": ("employment_services", "Service"),
    "Fitness Center": ("fitness_center", "Fitness"),
    "Florist": ("florist", "Retail"),
    "Furniture": ("furniture", "Retail"),
    "Gas Station": ("gas_station", "Service"),
    "Gift Shop": ("gift_shop", "Retail"),
    "Grocery": ("grocery", "Groceries"),
    "Home Decor": ("home_decor", "Retail"),
    "Home Services": ("home_services", "Service"),
    "Hospital": ("hospital", "Healthcare"),
    "Hotel": ("hotel", "Service"),
    "Jewelry Repair": ("jewelry_repair", "Service"),
    "Jewelry Store": ("jewelry_store", "Retail"),
    "Job Training": ("job_training", "Education"),
    "Legal": ("legal", "Service"),
    "Library": ("library", "Civic & Religion"),
    "Liquor Store": ("liquor_store", "Retail"),
    "Meat/Fish Market": ("meat_fish_market", "Groceries"),
    "Museum": ("museum", "Culture"),
    "Nightlife": ("nightlife", "Dining"),
    "Optician": ("optician", "Retail"),
    "Other Entertainment": ("other_entertainment", "Retail"),
    "Other Hospital": ("other_hospital", "Healthcare"),
    "Other Retail": ("other_retail", "Retail"),
    "Other School": ("other_school", "Education"),
    "Recreation Area": ("recreation_area", "Park"),
    "Restaurant": ("restaurant", "Dining"),
    "School Elem./Secd.": ("school_elem_secd", "Education"),
    "Shoes": ("shoes", "Retail"),
    "Shopping Center": ("shopping_center", "Retail"),
    "Social Services": ("social_services", "Service"),
    "Specific Sports": ("specific_sports", "Fitness"),
    "Sports Center": ("sports_center", "Fitness"),
    "Sports Equipment": ("sports_equipment", "Retail"),
    "Stadium": ("stadium", "Civic & Religion"),
    "Surgical Hospital": ("surgical_hospital", "Healthcare"),
    "Tire Shops": ("tire_shops", "Service"),
    "Tourist Attraction": ("tourist_attraction", "Tourism"),
    "Tourist Info": ("tourist_info", "Tourism"),
    "Toy Store": ("toy_store", "Retail"),
    "Train Station": ("train_station", "Transport"),
    "Travel Agency": ("travel_agency", "Tourism"),
    "Zoo & Botanical Garden": ("zoo___botanical_garden", "Culture"),
}

POI_COUNT_EDGES = (1, 500, 2_000, 5_000, np.inf)
POI_MARKER_AREAS = {"1–499": 16, "500–1,999": 32, "2,000–4,999": 56, "5,000+": 90}
POI_IDENTITY = ["Class", "NAME", "LATITUDE", "LONGITUDE"]


def trimmed_mean(values):
    """Average all values within the inclusive sample quantile bounds."""
    values = pd.Series(values)
    lo, hi = values.quantile([0.25, 0.75])
    return values[values.between(lo, hi)].mean()


def home_centres(cells):
    """Return H3 home centres in the Finnish national metric CRS."""
    centres = [h3.cell_to_latlng(cell) for cell in cells]
    lat, lon = np.asarray(centres, dtype=float).reshape(-1, 2).T
    return gpd.GeoDataFrame(
        {"home_gid9": cells}, geometry=gpd.points_from_xy(lon, lat), crs=4326
    ).to_crs(3067)


def match_pois_to_cells(pois, visited_cells):
    """Intersect projected POI points with the study's H3 polygons."""
    polygons = [
        Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(cell)])
        for cell in visited_cells
    ]
    grid = gpd.GeoDataFrame({"stay_gid9": visited_cells}, geometry=polygons, crs=4326).to_crs(3067)
    return gpd.sjoin(pois[["Class", "x", "y", "geometry"]], grid, predicate="intersects")


def build_summary_df(places, pois, study_areas):
    """Summarise paired distances from K-place rows with a `stay_gid9` column."""
    recurrent = places.loc[places.k_freq.eq(1) & places.visit_freq.gt(0)]
    homes = home_centres(recurrent.home_gid9.unique())
    study_homes = gpd.sjoin(
        homes, study_areas[["geometry"]].to_crs(3067), predicate="intersects"
    ).home_gid9
    recurrent = recurrent.loc[recurrent.home_gid9.isin(study_homes)].sort_values(
        "visit_freq", ascending=False, kind="stable"
    )
    home_points = homes.set_index("home_gid9").geometry
    home_coordinates = pd.DataFrame({"x": home_points.x, "y": home_points.y})

    # Exclude Park and Tourism from the amenity distance analysis.
    included_classes = [
        original for original, (_, category) in AMENITY_CLASSES.items()
        if category not in {"Park", "Tourism"}
    ]
    pois = pois.loc[pois["Class"].isin(included_classes)].dropna(subset=["geometry"])
    counts = pois.drop_duplicates(POI_IDENTITY).groupby("Class").size()
    pois = pois.to_crs(3067)
    pois = pois.assign(x=pois.geometry.x, y=pois.geometry.y)
    points_by_class = pois.groupby("Class")[["x", "y"]]
    memberships = match_pois_to_cells(pois, recurrent.stay_gid9.unique())

    rows = []
    for original_class, class_pois in memberships.groupby("Class", observed=True):
        selected = recurrent.loc[recurrent.stay_gid9.isin(class_pois.stay_gid9)].drop_duplicates(
            "user_id"
        )
        home_xy = home_coordinates.loc[selected.home_gid9].to_numpy()
        nearest = cKDTree(points_by_class.get_group(original_class)).query(home_xy)[0]
        recurrent_distances = np.empty(len(selected))
        pois_by_cell = class_pois.groupby("stay_gid9", sort=False)[["x", "y"]]
        for cell, positions in selected.groupby("stay_gid9", sort=False).indices.items():
            cell_tree = cKDTree(pois_by_cell.get_group(cell))
            recurrent_distances[positions] = cell_tree.query(home_xy[positions])[0]
        amenity, category = AMENITY_CLASSES[original_class]
        rows.append({
            "amenity": amenity,
            "original_class": original_class,
            "category": category,
            "n_users": len(selected),
            "d_freq": trimmed_mean(recurrent_distances),
            "d_prox": trimmed_mean(nearest),
        })

    summary = pd.DataFrame(
        rows, columns=["amenity", "original_class", "category", "n_users", "d_freq", "d_prox"]
    ).sort_values(["category", "amenity"], ignore_index=True)
    
    summary["delta_d"] = summary.d_freq - summary.d_prox
    summary["delta_d_rel"] = summary.delta_d / summary.d_prox  
    
    summary["n_pois"] = summary.original_class.map(counts)
    summary["size_bin"] = pd.cut(
        summary.n_pois, POI_COUNT_EDGES, labels=list(POI_MARKER_AREAS), right=False
    )
    summary["area_pt2"] = summary.size_bin.map(POI_MARKER_AREAS).astype(float)
    return summary
