import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


REVERSE_CLASS_MAPPING = {
    "airport": "Airport",
    "amusement": "Amusement",
    "appliances_store": "Appliances Store",
    "auto_service": "Auto Service",
    "auto_home_supply": "Auto/Home Supply",
    "bakery": "Bakery",
    "barber": "Barber",
    "beach": "Beach",
    "beauty_salon": "Beauty Salon",
    "bike___motorcycle_parking": "Bike & Motorcycle Parking",
    "book_store": "Book Store",
    "bus_stop": "Bus Stop",
    "cafe": "Cafe",
    "car_dealer": "Car Dealer",
    "car_rental": "Car Rental",
    "car_wash": "Car Wash",
    "casino": "Casino",
    "cemetery": "Cemetery",
    "cinema": "Cinema",
    "clinic": "Clinic",
    "clothing___accessories": "Clothing & Accessories",
    "concert_hall": "Concert Hall",
    "confectionery": "Confectionery",
    "convention_center": "Convention Center",
    "cultural_center": "Cultural Center",
    "dairy_store": "Dairy Store",
    "daycare": "Daycare",
    "dentist": "Dentist",
    "electrical_repair": "Electrical Repair",
    "electronics_repair": "Electronics Repair",
    "employment_services": "Employment Services",
    "fitness_center": "Fitness Center",
    "florist": "Florist",
    "furniture": "Furniture",
    "gas_station": "Gas Station",
    "gift_shop": "Gift Shop",
    "grocery": "Grocery",
    "home_decor": "Home Decor",
    "home_services": "Home Services",
    "hospital": "Hospital",
    "hotel": "Hotel",
    "jewelry_repair": "Jewelry Repair",
    "jewelry_store": "Jewelry Store",
    "job_training": "Job Training",
    "legal": "Legal",
    "library": "Library",
    "liquor_store": "Liquor Store",
    "meat_fish_market": "Meat/Fish Market",
    "museum": "Museum",
    "nightlife": "Nightlife",
    "optician": "Optician",
    "other_entertainment": "Other Entertainment",
    "other_hospital": "Other Hospital",
    "other_retail": "Other Retail",
    "other_school": "Other School",
    "parking": "Parking",
    "psychiatric_hospital": "Psychiatric Hospital",
    "race_track": "Race Track",
    "recreation_area": "Recreation Area",
    "restaurant": "Restaurant",
    "school_elem_secd": "School Elem./Secd.",
    "shoes": "Shoes",
    "shopping_center": "Shopping Center",
    "social_services": "Social Services",
    "specific_sports": "Specific Sports",
    "sports_center": "Sports Center",
    "sports_equipment": "Sports Equipment",
    "stadium": "Stadium",
    "surgical_hospital": "Surgical Hospital",
    "taxi_stand": "Taxi Stand",
    "tire_shops": "Tire Shops",
    "tourist_attraction": "Tourist Attraction",
    "tourist_info": "Tourist Info",
    "toy_store": "Toy Store",
    "train_station": "Train Station",
    "travel_agency": "Travel Agency",
    "wildlife_park": "Wildlife Park",
    "zoo___botanical_garden": "Zoo & Botanical Garden",
}

CLASS_TO_CATEGORY_MAPPING = {
    "Airport": "Transport",
    "Amusement": "Culture",
    "Appliances Store": "Retail",
    "Auto Service": "Service",
    "Auto/Home Supply": "Retail",
    "Bakery": "Groceries",
    "Barber": "Service",
    "Beach": "Park",
    "Beauty Salon": "Service",
    "Bike & Motorcycle Parking": "Transport",
    "Book Store": "Retail",
    "Bus Stop": "Transport",
    "Cafe": "Dining",
    "Car Dealer": "Retail",
    "Car Rental": "Retail",
    "Car Wash": "Service",
    "Casino": "Retail",
    "Cemetery": "Civic & Religion",
    "Cinema": "Culture",
    "Clinic": "Healthcare",
    "Clothing & Accessories": "Retail",
    "Concert Hall": "Culture",
    "Confectionery": "Groceries",
    "Convention Center": "Civic & Religion",
    "Cultural Center": "Culture",
    "Dairy Store": "Groceries",
    "Daycare": "Education",
    "Dentist": "Healthcare",
    "Electrical Repair": "Service",
    "Electronics Repair": "Service",
    "Employment Services": "Service",
    "Fitness Center": "Fitness",
    "Florist": "Retail",
    "Furniture": "Retail",
    "Gas Station": "Service",
    "Gift Shop": "Retail",
    "Grocery": "Groceries",
    "Home Decor": "Retail",
    "Home Services": "Service",
    "Hospital": "Healthcare",
    "Hotel": "Service",
    "Jewelry Repair": "Service",
    "Jewelry Store": "Retail",
    "Job Training": "Education",
    "Legal": "Service",
    "Library": "Civic & Religion",
    "Liquor Store": "Retail",
    "Meat/Fish Market": "Groceries",
    "Museum": "Culture",
    "Nightlife": "Dining",
    "Optician": "Retail",
    "Other Entertainment": "Retail",
    "Other Hospital": "Healthcare",
    "Other Retail": "Retail",
    "Other School": "Education",
    "Parking": "Transport",
    "Psychiatric Hospital": "Healthcare",
    "Race Track": "Fitness",
    "Recreation Area": "Park",
    "Restaurant": "Dining",
    "School Elem./Secd.": "Education",
    "Shoes": "Retail",
    "Shopping Center": "Retail",
    "Social Services": "Service",
    "Specific Sports": "Fitness",
    "Sports Center": "Fitness",
    "Sports Equipment": "Retail",
    "Stadium": "Civic & Religion",
    "Surgical Hospital": "Healthcare",
    "Taxi Stand": "Transport",
    "Tire Shops": "Service",
    "Tourist Attraction": "Tourism",
    "Tourist Info": "Service",
    "Toy Store": "Retail",
    "Train Station": "Transport",
    "Travel Agency": "Service",
    "Wildlife Park": "Park",
    "Zoo & Botanical Garden": "Culture",
}

CLASSES_TO_REMOVE = {
    "Race Track",
    "Wildlife Park",
    "Dairy Store",
    "Bike & Motorcycle Parking",
    "Confectionery",
    "Psychiatric Hospital",
    "Taxi Stand",
    "Electronics Repair",
    "Cemetery",
}

FIGURE_CLASSES_TO_REMOVE = CLASSES_TO_REMOVE | {
    "Tourist Attraction",
    "Tourist Info",
    "Travel Agency",
}


def trimmed_mean(values, lower_q=0.25, upper_q=0.75):
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    q_low = clean.quantile(lower_q)
    q_high = clean.quantile(upper_q)
    trimmed = clean[(clean >= q_low) & (clean <= q_high)]
    if trimmed.empty:
        return np.nan
    return float(trimmed.mean())


def nearest_available_distances(
    home_points,
    poi_points,
    *,
    home_col="home_gid9",
    class_col="amenity",
    x_col="x",
    y_col="y",
):
    """Find the nearest available POI of each class for every home.

    ``home_points`` and ``poi_points`` must use the same metric coordinate
    reference system. The result is a long table with one ``d_prox`` value for
    every home and amenity class. No observed visitation is used in this step.
    """
    home_required = [home_col, x_col, y_col]
    poi_required = [class_col, x_col, y_col]
    _require_columns(home_points, home_required, "home_points")
    _require_columns(poi_points, poi_required, "poi_points")

    homes = home_points[home_required].drop_duplicates(home_col).copy()
    pois = poi_points[poi_required].copy()
    for frame in (homes, pois):
        frame[x_col] = pd.to_numeric(frame[x_col], errors="coerce")
        frame[y_col] = pd.to_numeric(frame[y_col], errors="coerce")
        frame.dropna(subset=[x_col, y_col], inplace=True)

    if homes.empty:
        raise ValueError("home_points contains no valid metric coordinates")
    if pois.empty:
        raise ValueError("poi_points contains no valid metric coordinates")

    home_xy = homes[[x_col, y_col]].to_numpy(dtype=float)
    results = []
    for amenity, class_points in pois.dropna(subset=[class_col]).groupby(
        class_col, sort=True
    ):
        distances, _ = cKDTree(
            class_points[[x_col, y_col]].to_numpy(dtype=float)
        ).query(home_xy, k=1)
        results.append(
            pd.DataFrame(
                {
                    home_col: homes[home_col].to_numpy(),
                    "amenity": amenity,
                    "d_prox": distances,
                }
            )
        )

    if not results:
        raise ValueError("poi_points contains no non-missing amenity classes")
    return pd.concat(results, ignore_index=True)


def build_summary_df(
    places_k,
    grid_poi_classes,
    nearest_available,
    *,
    class_poi_counts=None,
    max_home_dist=50_000,
    display_offset=10.0,
):
    """Build the current amenity-level distance-differential summary.

    The calculation follows the Figure 5 convention:

    * ``d_freq`` is first calculated per user and amenity as the
      visit-frequency-weighted mean home distance across all ``K_freq`` places
      containing that amenity.
    * ``d_prox`` is the nearest available POI distance for the same user's home,
      irrespective of whether the user visited that POI.
    * The two user-level series are independently 25--75% trimmed and averaged.
    * ``delta_d_rel`` reproduces the committed plotting cache: the absolute
      aggregate differential after the 10 m display offset, divided by
      ``d_prox``. The signed, unshifted value is retained as ``delta_d_signed``.

    All inputs are in-memory tables. Callers may include home as an eligible
    ``K_freq`` supply point before calling this function, matching the current
    analysis convention.
    """
    place_columns = [
        "user_id",
        "stay_gid10",
        "home_gid9",
        "home_dist",
        "visit_freq",
        "k_freq",
    ]
    _require_columns(places_k, place_columns, "places_k")
    _require_columns(grid_poi_classes, ["stay_gid10"], "grid_poi_classes")
    _require_columns(
        nearest_available,
        ["home_gid9", "amenity", "d_prox"],
        "nearest_available",
    )

    class_columns = [
        column for column in grid_poi_classes.columns if column != "stay_gid10"
    ]
    if not class_columns:
        raise ValueError("grid_poi_classes contains no amenity-class columns")

    places = places_k[place_columns].drop_duplicates(
        ["user_id", "stay_gid10"]
    )
    homes_per_user = places.groupby("user_id")["home_gid9"].nunique(dropna=False)
    if homes_per_user.gt(1).any():
        raise ValueError("Each user must map to exactly one home_gid9")
    class_counts = grid_poi_classes[["stay_gid10", *class_columns]].copy()
    class_counts[class_columns] = (
        class_counts[class_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .clip(lower=0)
    )
    places = places.merge(class_counts, on="stay_gid10", how="left")
    places[class_columns] = places[class_columns].fillna(0)
    places["home_dist"] = pd.to_numeric(places["home_dist"], errors="coerce")
    places["visit_freq"] = pd.to_numeric(places["visit_freq"], errors="coerce")
    places = places[
        places["home_dist"].notna()
        & places["visit_freq"].gt(0)
        & places["home_dist"].ge(0)
    ].copy()
    if max_home_dist is not None:
        places = places[places["home_dist"] <= max_home_dist].copy()

    proximity = nearest_available[["home_gid9", "amenity", "d_prox"]].copy()
    proximity["d_prox"] = pd.to_numeric(proximity["d_prox"], errors="coerce")
    proximity = proximity[
        proximity["d_prox"].notna() & proximity["d_prox"].gt(0)
    ].drop_duplicates(["home_gid9", "amenity"])
    proximity_lookup = proximity.set_index(["home_gid9", "amenity"])["d_prox"]

    poi_count_lookup = _normalise_class_poi_counts(
        class_poi_counts, class_counts, class_columns
    )
    kfreq = places[places["k_freq"].eq(1)].copy()
    summary_rows = []

    for amenity in class_columns:
        supplied = kfreq[kfreq[amenity] > 0].copy()
        if supplied.empty:
            user_distances = pd.DataFrame(columns=["d_freq", "d_prox"])
        else:
            weights = supplied.groupby("user_id")["visit_freq"].sum()
            d_freq = (
                (supplied["home_dist"] * supplied["visit_freq"])
                .groupby(supplied["user_id"])
                .sum()
                .div(weights)
                .rename("d_freq")
            )
            user_homes = supplied.drop_duplicates("user_id").set_index("user_id")[
                "home_gid9"
            ]
            keys = pd.MultiIndex.from_arrays(
                [user_homes.to_numpy(), np.repeat(amenity, len(user_homes))],
                names=["home_gid9", "amenity"],
            )
            d_prox = pd.Series(
                proximity_lookup.reindex(keys).to_numpy(),
                index=user_homes.index,
                name="d_prox",
            )
            user_distances = pd.concat([d_freq, d_prox], axis=1).dropna()

        mean_d_freq = trimmed_mean(user_distances["d_freq"])
        mean_d_prox = trimmed_mean(user_distances["d_prox"])
        delta_signed = mean_d_freq - mean_d_prox
        delta_display = delta_signed + display_offset
        delta_relative = (
            abs(delta_display) / mean_d_prox
            if pd.notna(mean_d_prox) and mean_d_prox > 0
            else np.nan
        )
        summary_rows.append(
            {
                "amenity": amenity,
                "original_class": REVERSE_CLASS_MAPPING.get(amenity),
                "category": CLASS_TO_CATEGORY_MAPPING.get(
                    REVERSE_CLASS_MAPPING.get(amenity)
                ),
                "d_freq": mean_d_freq,
                "d_prox": mean_d_prox,
                "delta_d_signed": delta_signed,
                "delta_d_display": delta_display,
                "delta_d_rel": delta_relative,
                "poi_count": int(poi_count_lookup.get(amenity, 0)),
                "n_users": int(len(user_distances)),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary = summary[~summary["original_class"].isin(CLASSES_TO_REMOVE)]
    return summary.reset_index(drop=True)


def figure_cache_columns(summary_df):
    """Return the privacy-safe columns committed for Figure 5 plotting."""
    required = [
        "amenity",
        "original_class",
        "category",
        "d_prox",
        "delta_d_rel",
        "poi_count",
    ]
    _require_columns(summary_df, required, "summary_df")
    public = summary_df[~summary_df["original_class"].isin(FIGURE_CLASSES_TO_REMOVE)]
    return public[required].reset_index(drop=True)


def _normalise_class_poi_counts(class_poi_counts, class_counts, class_columns):
    if class_poi_counts is None:
        return class_counts[class_columns].sum(axis=0)
    if isinstance(class_poi_counts, pd.Series):
        return pd.to_numeric(class_poi_counts, errors="coerce").fillna(0)
    _require_columns(class_poi_counts, ["amenity", "poi_count"], "class_poi_counts")
    return (
        class_poi_counts.drop_duplicates("amenity")
        .set_index("amenity")["poi_count"]
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
    )


def _require_columns(frame, required, frame_name):
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")
