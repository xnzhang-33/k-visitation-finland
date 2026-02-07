import numpy as np
import pandas as pd


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


def build_summary_df(
    places_k,
    grid_poi_classes,
    max_home_dist=50_000,
):
    """
    Build amenity-level distance summary from in-memory DataFrames.
    No file/path IO is used in this function.
    """
    places_k = places_k.drop_duplicates(subset=["user_id", "stay_gid10"]).copy()
    poi_class_columns = [col for col in grid_poi_classes.columns if col != "stay_gid10"]

    places_with_poi = places_k.merge(grid_poi_classes, on="stay_gid10", how="left")
    places_with_poi[poi_class_columns] = places_with_poi[poi_class_columns].fillna(0)

    kfreq_places = places_with_poi[places_with_poi["k_freq"] == 1].copy()

    user_kfreq_reference = []
    for user_id, user_places in kfreq_places.groupby("user_id"):
        user_places_sorted = user_places.sort_values("visit_freq", ascending=False)
        row = {"user_id": user_id}
        for poi_class in poi_class_columns:
            places_with_amenity = user_places_sorted[user_places_sorted[poi_class] > 0]
            row[f"{poi_class}_kfreq_dist"] = (
                places_with_amenity.iloc[0]["home_dist"] if len(places_with_amenity) > 0 else np.nan
            )
        user_kfreq_reference.append(row)
    kfreq_reference_df = pd.DataFrame(user_kfreq_reference)

    user_nearest = []
    for user_id, user_places in places_with_poi.groupby("user_id"):
        row = {"user_id": user_id}
        for poi_class in poi_class_columns:
            places_with_amenity = user_places[user_places[poi_class] > 0]
            row[f"{poi_class}_nearest_dist"] = (
                places_with_amenity.loc[places_with_amenity["home_dist"].idxmin(), "home_dist"]
                if len(places_with_amenity) > 0
                else np.nan
            )
        user_nearest.append(row)
    nearest_df = pd.DataFrame(user_nearest)

    combined_distances = kfreq_reference_df.merge(nearest_df, on="user_id", how="outer")

    summary_rows = []
    for amenity in poi_class_columns:
        kfreq_col = f"{amenity}_kfreq_dist"
        nearest_col = f"{amenity}_nearest_dist"

        if kfreq_col in combined_distances.columns:
            mean_kfreq_dist = trimmed_mean(combined_distances[kfreq_col])
            valid_idx = combined_distances[kfreq_col].notna()
        else:
            mean_kfreq_dist = np.nan
            valid_idx = pd.Series(True, index=combined_distances.index)

        mean_closest_visit_dist = (
            trimmed_mean(combined_distances.loc[valid_idx, nearest_col])
            if nearest_col in combined_distances.columns
            else np.nan
        )

        n_places = int((places_with_poi[amenity] > 0).sum())

        summary_rows.append(
            {
                "amenity": amenity,
                "mean_kfreq_dist": mean_kfreq_dist,
                "mean_closest_visit_dist": mean_closest_visit_dist,
                "n_places": n_places,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df["dist_diff"] = summary_df["mean_kfreq_dist"] - summary_df["mean_closest_visit_dist"] + 10 # Add small constant to avoid zero values, because we use log scale in the plot

    # Remap class names to original and assign categories
    summary_df["original_class"] = summary_df["amenity"].map(REVERSE_CLASS_MAPPING)
    summary_df["category"] = summary_df["original_class"].map(CLASS_TO_CATEGORY_MAPPING)
    summary_df = summary_df[~summary_df["original_class"].isin(CLASSES_TO_REMOVE)]
    

    return summary_df.reset_index(drop=True)
