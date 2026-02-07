import numpy as np
import pandas as pd
from scipy.stats import linregress
from tqdm import tqdm
import h3

# Unified exploration decay rate across simulations
GAMMA_DEFAULT = 0.23

# --- 0. Generate simulated mobility patterns
def generate_mock_data(n_users=50, n_locations=300, random_seed=33):
   """Generate consistent mock data for testing."""
   
   # 重设随机种子以确保可重复性
   if random_seed is not None:
       np.random.seed(random_seed)
   
   # 0. Define Amenities first
   amenities = [
    'CIVIC_RELIGION', 'CULTURE', 'DINING', 'EDUCATION', 'FITNESS', 
    'GROCERIES', 'HEALTHCARE', 'RETAIL', 'SERVICE', 'TRANSPORT'
   ]
   
   # 1. Generate Mock City Grid (Locations)
   # We simulate H3 indices with integers for simplicity in this demo, or use random strings
   loc_ids = [f"{i:03d}" for i in range(n_locations)]
   
   # Random coordinates centered around Helsinki roughly
   lat_center, lng_center = 60.1699, 24.9384
   lats = lat_center + np.random.normal(0, 0.05, n_locations)
   lngs = lng_center + np.random.normal(0, 0.05, n_locations)
   
   # Random POI counts (Log-normal distribution)
   poi_counts = np.random.lognormal(mean=2, sigma=1, size=n_locations).astype(int)
   poi_counts = np.maximum(poi_counts, 1)
   
   city_grid = pd.DataFrame({
       'h3_index': loc_ids,
       'lat': lats,
       'lng': lngs,
       'poi_count': poi_counts,
       'log_poi': np.log1p(poi_counts)
   })
   
   # Assign amenities to locations (Grid-level)
   # This ensures that if multiple users visit the same location, it has the same amenities
   for am in amenities:
       city_grid[am] = np.random.randint(0, 2, size=n_locations)
   
   # 2. Generate Mock User Homes
   user_ids = [f"user_{i:03d}" for i in range(n_users)]
   
   # Assign each user to a random "home" location from the grid
   home_locs = np.random.choice(loc_ids, size=n_users)
   user_home_lookup = pd.DataFrame({
       'user_id': user_ids,
       'home_gid9': home_locs 
   })
   
   # 3. Generate Mock Visitation Data (Stays)
   visits = []
   
   for user, home in zip(user_ids, home_locs):
       # User visits 10-100 locations
       n_visits = np.random.randint(10, 100)
       
       # Get home coordinates
       home_rec = city_grid.loc[city_grid['h3_index']==home].iloc[0]
       home_lat = home_rec['lat']
       home_lng = home_rec['lng']
       
       # Calculate distances using vectorized haversine from mobility_utils
       dists = haversine_distance_vectorized(
           home_lat, home_lng,
           city_grid['lat'].values,
           city_grid['lng'].values
       )
       
       # Generate probabilities (Distance decay - Gravity model style)
       # Add small buffer to dist to avoid division by zero or extreme peaks
       # dists is in meters (likely), or whatever haversine returns. 
       # Assuming mu matches units. Usually meters.
       probs = 1 / (dists + 100)**2 
       probs /= probs.sum()
       
       # Select locations
       chosen_indices = np.random.choice(len(city_grid), size=n_visits, p=probs, replace=True)
       
       # Create stay records
       for i, idx in enumerate(chosen_indices):
           loc_info = city_grid.iloc[idx]
           dist_m = dists[idx] # Corresponding distance
           
           visits.append({
               'user_id': user,
               'stay_gid10': loc_info['h3_index'],
               'start': pd.Timestamp('2023-01-01') + pd.Timedelta(days=i), 
               'visit_freq': 1,
               'home_dist': dist_m
           })
           
   df_visitation = pd.DataFrame(visits)
   
   # Merge amenities from city_grid to visitation data based on location
   # This makes sure the amenities in visitation data match the "real" properties of the location
   df_visitation = df_visitation.merge(
       city_grid[['h3_index'] + amenities],
       left_on='stay_gid10',
       right_on='h3_index',
       how='left'
   ).drop(columns=['h3_index'])
   
   return city_grid, user_home_lookup, df_visitation, amenities



# --- 1. Gravity & Distance Functions ---

def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    """
    Calculate Haversine distance in km between points (vectorized).
    """
    R = 6371.0  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def compute_gravity_weights(locations_df):
    """
    Compute gravity-based interaction weights for all location pairs.
    
    Cost function: C_ij = (d_i * d_j) / R_ij^2
    where d_i, d_j are POI counts (or other attraction metrics) and R_ij is distance.
    
    Parameters:
    -----------
    locations_df : pd.DataFrame
        Must contain columns: ['lat', 'lng', 'log_poi', 'h3_index']
    
    Returns: 
    --------
    weight_df : pd.DataFrame
        Normalized weight matrix where rows sum to 1.
    """
    n_locs = len(locations_df)
    
    # Extract coordinates and POI counts
    lats = locations_df['lat'].values
    lngs = locations_df['lng'].values
    log_pois = locations_df['log_poi'].values
    h3_indices = locations_df['h3_index'].values
    
    # Normalize opportunity density so weights are comparable across grids
    poi_sum = log_pois.sum()
    if poi_sum > 0:
        log_pois = log_pois / poi_sum
    
    # Compute pairwise distances
    dist_matrix = np.zeros((n_locs, n_locs))
    for i in range(n_locs):
        dist_matrix[i, :] = haversine_distance_vectorized(
            lats[i], lngs[i], lats, lngs
        )
    
    # Avoid division by zero
    dist_matrix[dist_matrix < 0.1] = 0.1
    
    # POI product matrix
    poi_product = np.outer(log_pois, log_pois)
    
    # Gravity cost function: C_ij = (d_i * d_j) / R_ij^2
    cost_matrix = (poi_product) / (dist_matrix ** 2)
    
    # Normalize each row to get probability weights
    row_sums = cost_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-10
    
    weight_matrix = cost_matrix / row_sums
    
    # Convert to DataFrame for easier indexing
    weight_df = pd.DataFrame(
        weight_matrix,
        index=h3_indices,
        columns=h3_indices
    )
    
    return weight_df


# --- 2. EPR Calibration Functions ---

def calibrate_rho_params(df_stays, sampled_users=None, min_S=1, log_bin=True, n_log_bins=50):
    """
    Calibrate EPR exploration parameters (rho, gamma) from data.
    P_new(S) = rho * S^{-gamma}
    """
    if sampled_users is None:
        sampled_users = df_stays['user_id'].unique()

    # Handle Dask vs Pandas
    if hasattr(df_stays, 'compute'):
        df = df_stays[df_stays["user_id"].isin(sampled_users)][["user_id", "start", "stay_gid10"]].compute()
    else:
        df = df_stays[df_stays["user_id"].isin(sampled_users)][["user_id", "start", "stay_gid10"]].copy()

    df = df.sort_values(["user_id", "start"]).reset_index(drop=True)

    # --- Compute y_t (new location indicator) ---
    df["is_new_loc"] = (~df.duplicated(["user_id", "stay_gid10"])).astype(np.int8)
    df["S_after"] = df.groupby("user_id")["is_new_loc"].cumsum().astype(np.int32)
    df["S_prev"] = df.groupby("user_id")["S_after"].shift(1).fillna(0).astype(np.int32)

    df = df[df["S_prev"] >= min_S].copy()

    if df.empty:
        return np.nan, np.nan, pd.DataFrame()

    # --- Estimate empirical P_new(S) ---
    if not log_bin:
        g = df.groupby("S_prev")["is_new_loc"].agg(["mean", "count"]).reset_index()
        calib_df = g.rename(columns={"S_prev": "S", "mean": "p_new"})
    else:
        # Log-binning S
        S_vals = df["S_prev"].to_numpy()
        y_vals = df["is_new_loc"].to_numpy()

        S_min = max(min_S, int(S_vals.min()))
        S_max = int(S_vals.max())
        
        if S_min >= S_max:
             return np.nan, np.nan, pd.DataFrame()

        edges = np.unique(np.floor(np.logspace(np.log10(S_min), np.log10(S_max + 1), n_log_bins)).astype(int))
        edges = np.clip(edges, S_min, S_max + 1)
        edges = np.unique(np.r_[edges, S_max + 1])

        bin_id = np.digitize(S_vals, edges, right=False) - 1
        calib_rows = []

        for b in range(len(edges) - 1):
            mask = (bin_id == b)
            cnt = int(mask.sum())
            if cnt == 0: continue

            p = float(y_vals[mask].mean())
            S_mid = float(np.exp(np.mean(np.log(S_vals[mask]))))
            calib_rows.append({"S": S_mid, "p_new": p, "count": cnt})

        calib_df = pd.DataFrame(calib_rows).sort_values("S").reset_index(drop=True)

    calib_df = calib_df[(calib_df["p_new"] > 0) & np.isfinite(calib_df["p_new"])]
    if len(calib_df) < 3:
        return np.nan, np.nan, calib_df

    # --- Fit poly ---
    x = np.log(calib_df["S"].to_numpy())
    y = np.log(calib_df["p_new"].to_numpy())
    w = np.sqrt(calib_df["count"].to_numpy().astype(float)) 

    slope, intercept = np.polyfit(x, y, deg=1, w=w)

    gamma_hat = -slope
    rho_hat = float(np.exp(intercept))
    rho_hat = float(np.clip(rho_hat, 1e-12, 1.0))

    calib_df["p_fit"] = rho_hat * (calib_df["S"] ** (-gamma_hat))

    return rho_hat, gamma_hat, calib_df

def calculate_user_specific_rho(df_stays, user_id, gamma=GAMMA_DEFAULT, min_S=1):
    """
    Calculate user-specific rho given a global gamma.
    """
    user_trace = df_stays[df_stays['user_id'] == user_id].sort_values('start').copy()
    
    if len(user_trace) < 5: return np.nan
    
    user_trace['is_new_loc'] = (~user_trace.duplicated(['user_id', 'stay_gid10'])).astype(np.int8)
    user_trace['S_after'] = user_trace['is_new_loc'].cumsum().astype(np.int32)
    user_trace['S_prev'] = user_trace['S_after'].shift(1).fillna(0).astype(np.int32)
    user_trace = user_trace[user_trace['S_prev'] >= min_S].copy()
    
    if len(user_trace) < 3: return np.nan
    
    calib_df = user_trace.groupby('S_prev')['is_new_loc'].agg(['mean', 'count']).reset_index()
    calib_df = calib_df.rename(columns={'S_prev': 'S', 'mean': 'p_new'})
    calib_df = calib_df[(calib_df['p_new'] > 0)]
    
    if len(calib_df) < 2: return np.nan
    
    x = np.log(calib_df['S'].to_numpy())
    y = np.log(calib_df['p_new'].to_numpy())
    w = np.sqrt(calib_df['count'].to_numpy())
    
    log_rho_estimates = y + gamma * x
    log_rho = np.average(log_rho_estimates, weights=w)
    
    return np.clip(np.exp(log_rho), 1e-12, 1.0)


# --- 3. d-EPR Simulation Functions ---

def simulate_depr_for_grid(home_gid9, df_visitation, user_home_lookup, city_grid, 
                          user_rho_dict, max_steps=500):
    """
    Run d-EPR simulation for users in a specific home grid.
    """
    # Enforce unified gamma across simulations
    gamma = GAMMA_DEFAULT
    np.random.seed(42)

    users_in_grid = user_home_lookup[user_home_lookup['home_gid9'] == home_gid9]['user_id'].values
    if len(users_in_grid) == 0:
        return pd.DataFrame(columns=['user_id', 'stay_gid10', 'visit_freq_synth'])
    
    grid_visitation = df_visitation[df_visitation['user_id'].isin(users_in_grid)].copy()
    unique_visited_locations = grid_visitation['stay_gid10'].unique()
    
    if len(grid_visitation) == 0:
        return pd.DataFrame(columns=['user_id', 'stay_gid10', 'visit_freq_synth'])

    # Choose the most visited stay_gid10 within this home_gid9 (grid-wide)
    if 'visit_freq' in grid_visitation.columns:
        loc_counts = grid_visitation.groupby('stay_gid10')['visit_freq'].sum()
    else:
        loc_counts = grid_visitation.groupby('stay_gid10').size()
    grid_most_visited_loc = loc_counts.idxmax()
    
    location_catalog = city_grid[city_grid['h3_index'].isin(unique_visited_locations)].copy()
    
    if len(location_catalog) < 2:
        return pd.DataFrame(columns=['user_id', 'stay_gid10', 'visit_freq_synth'])
        
    gravity_weights = compute_gravity_weights(location_catalog)
    
    synthetic_visits = []
    
    for user_id in users_in_grid:
        user_visits = grid_visitation[grid_visitation['user_id'] == user_id]
        if len(user_visits) == 0: continue
        
        user_rho = user_rho_dict.get(user_id, 0.6) 
        
        # Initialize at the most visited stay_gid10 within this home_gid9
        home_loc = grid_most_visited_loc
        
        if home_loc not in gravity_weights.index:
            continue
            
        total_steps = min(max(user_visits['visit_freq'].sum(), 10), max_steps)
        
        visited = {home_loc: 1}
        current_loc = home_loc
        
        for step in range(total_steps):
            S = len(visited)
            
            P_new = min(user_rho * (S ** (-gamma)), 1.0)
            
            # Decide: Explore or Return
            if np.random.random() < P_new:
                # EXPLORE
                weights = gravity_weights.loc[current_loc].values
                candidates = gravity_weights.columns.values
                
                mask_unvisited = np.isin(candidates, list(visited.keys()), invert=True)
                
                if mask_unvisited.sum() > 0:
                    w_unvisited = weights[mask_unvisited]
                    if w_unvisited.sum() == 0:
                        w_unvisited = np.ones_like(w_unvisited)
                    
                    w_unvisited /= w_unvisited.sum()
                    
                    new_loc = np.random.choice(candidates[mask_unvisited], p=w_unvisited)
                    visited[new_loc] = visited.get(new_loc, 0) + 1
                    current_loc = new_loc
                    continue 

            # RETURN
            visit_locs = list(visited.keys())
            visit_counts = np.array([visited[k] for k in visit_locs])
            prob_return = visit_counts / visit_counts.sum()
            
            return_loc = np.random.choice(visit_locs, p=prob_return)
            visited[return_loc] += 1
            current_loc = return_loc
            
        for loc, count in visited.items():
            synthetic_visits.append({
                'user_id': user_id,
                'stay_gid10': loc,
                'visit_freq_synth': count
            })
            
    return pd.DataFrame(synthetic_visits)
