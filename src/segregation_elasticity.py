import numpy as np
import pandas as pd
from scipy import stats


def calculate_grid_elasticity(grid_data, q_k_col='qk', soc_int_col='sk', min_users=10):
    """Calculate elasticity using log-log regression: ε = ∂ln(S_K)/∂ln(q_K)"""
    
    valid_data = grid_data[
        (grid_data[q_k_col] > 0) & 
        (grid_data[soc_int_col] > 0) & 
        (grid_data[q_k_col].notna()) & 
        (grid_data[soc_int_col].notna())
    ].copy()
    
    if len(valid_data) < min_users:
        return {'es': np.nan, 'r2': np.nan, 'p': np.nan, 'n_users': len(valid_data), 'stderr': np.nan}
    
    log_qk = np.log1p(valid_data[q_k_col])
    log_soc = np.log1p(valid_data[soc_int_col])
    
    slope, r_value, p_value, std_err = stats.linregress(log_qk, log_soc)
    
    return {
        'es': slope,
        'r2': r_value ** 2,
        'p': p_value,
        'n_users': len(valid_data),
        'stderr': std_err
    }


def compute_es_by_grid(user_data, groupby_col='home_grd_id', q_k_col='qk', soc_int_col='sk', min_users=10):
    """
    Compute elasticity for each grid cell and return DataFrame
    
    Parameters:
    -----------
    user_data : pd.DataFrame
        User-level data containing grouping and variable columns
    groupby_col : str
        Grouping column name (default: 'home_grd_id')
    q_k_col : str
        K-value column name (default: 'qk')
    soc_int_col : str
        Social interaction column name (default: 'sk')
    min_users : int
        Minimum user count threshold (default: 10)
        
    Returns:
    --------
    pd.DataFrame
        Grid-level data containing elasticity and statistical measures
    """
    results = []
    
    for grd_id, group_data in user_data.groupby(groupby_col):
        if pd.isna(grd_id):
            continue
        
        result = calculate_grid_elasticity(group_data, q_k_col, soc_int_col, min_users)
        result['grd_id'] = grd_id
        results.append(result)
    
    return pd.DataFrame(results)