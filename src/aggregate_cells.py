"""Aggregate user metrics to H3 cells and then to supplied 1 km grid cells."""


def aggregate_to_h3(users, metrics, h3_col='home_gid9'):
    """Calculate user means and valid counts, with one supplied row per user."""
    grouped = users.groupby(h3_col, sort=False)
    means = grouped[metrics].mean()
    counts = grouped[metrics].count().add_suffix('_count')
    return means.join(counts).join(grouped.size().rename('n_users')).reset_index()


def aggregate_to_1km(cells, crosswalk, metrics, h3_col='home_gid9', grid_col='grd_id'):
    """Calculate equal-H3 means within each supplied 1 km grid cell."""
    mapped = cells.merge(crosswalk[[h3_col, grid_col]], on=h3_col, how='inner')
    grouped = mapped.groupby(grid_col, sort=False)
    means = grouped[metrics].mean()
    count_columns = [f'{metric}_count' for metric in metrics]
    counts = grouped[count_columns + ['n_users']].sum(min_count=1)
    return means.join(counts).reset_index()
