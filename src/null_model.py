"""Fit user exploration rates and simulate d-EPR visits."""

import numpy as np
import pandas as pd
from pyproj import Transformer


GAMMA_DEFAULT = 0.23


def estimate_user_rho(stays, user_id, gamma=GAMMA_DEFAULT, min_locations=1):
    """Fit rho from the user's ordered stays with gamma held fixed."""
    trace = stays.loc[stays['user_id'].eq(user_id)].sort_values('start')
    if len(trace) < 5:
        return np.nan

    is_new = (~trace['stay_gid10'].duplicated()).astype(np.int8)
    locations_seen = is_new.cumsum().shift(1, fill_value=0)
    eligible = locations_seen.ge(min_locations)
    if eligible.sum() < 3:
        return np.nan

    rates = is_new[eligible].groupby(locations_seen[eligible]).agg(['mean', 'count'])
    rates = rates.loc[rates['mean'].gt(0)]
    if len(rates) < 2:
        return np.nan

    log_rho = np.log(rates['mean'].to_numpy()) + gamma * np.log(rates.index.to_numpy())
    weights = np.sqrt(rates['count'].to_numpy())
    return np.clip(np.exp(np.average(log_rho, weights=weights)), 1e-12, 1.0)


def gravity_weights(locations):
    """Use EPSG:3067 Euclidean distances with a 100 m floor, then normalise."""
    project = Transformer.from_crs(4326, 3067, always_xy=True)
    x, y = project.transform(locations['lng'].to_numpy(), locations['lat'].to_numpy())
    distances = np.hypot(x[:, None] - x, y[:, None] - y)
    distances = np.maximum(distances, 100.0)

    density = locations['log_poi'].to_numpy()
    if density.sum() > 0:
        density = density / density.sum()
    weights = np.outer(density, density) / distances**2
    totals = weights.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1e-10
    cells = locations['h3_index']
    return pd.DataFrame(weights / totals, index=cells.to_numpy(), columns=cells.to_numpy())


def simulate_grid_visits(home_grid, stays, user_homes, city_grid, user_rho, max_steps=500, *, seed=42):
    """Simulate users in one home grid from its most visited location."""
    columns = ['user_id', 'stay_gid10', 'visit_freq_synth']
    rng = np.random.RandomState(seed)
    users = user_homes.loc[user_homes['home_gid9'].eq(home_grid), 'user_id']
    grid_stays = stays.loc[stays['user_id'].isin(users)]
    visits_by_cell = grid_stays.groupby('stay_gid10')
    counts = visits_by_cell['visit_freq'].sum() if 'visit_freq' in grid_stays else visits_by_cell.size()
    locations = city_grid.loc[city_grid['h3_index'].isin(counts.index)]
    if len(locations) < 2:
        return pd.DataFrame(columns=columns)

    start = counts.idxmax()
    weights = gravity_weights(locations)
    if start not in weights.index:
        return pd.DataFrame(columns=columns)

    candidates = weights.columns.to_numpy()
    visits = []
    for user in users:
        user_stays = grid_stays.loc[grid_stays['user_id'].eq(user)]
        if user_stays.empty:
            continue
        steps = min(max(user_stays['visit_freq'].sum(), 10), max_steps)
        rho = user_rho.get(user, 0.6)
        current = start
        visited = {start: 1}

        for _ in range(steps):
            if rng.random_sample() < min(rho * len(visited)**(-GAMMA_DEFAULT), 1.0):
                unvisited = ~np.isin(candidates, list(visited))
                if unvisited.any():
                    probabilities = weights.loc[current].to_numpy()[unvisited]
                    if probabilities.sum() == 0:
                        probabilities = np.ones_like(probabilities)
                    current = rng.choice(candidates[unvisited], p=probabilities / probabilities.sum())
                    visited[current] = 1
                    continue

            visit_counts = np.array(list(visited.values()))
            current = rng.choice(list(visited), p=visit_counts / visit_counts.sum())
            visited[current] += 1

        visits.extend((user, cell, count) for cell, count in visited.items())

    return pd.DataFrame(visits, columns=columns)
