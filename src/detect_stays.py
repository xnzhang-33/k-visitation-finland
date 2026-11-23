"""Small-data strategy for stays and monthly home/work anchors."""

import h3
import pandas as pd
from infostop import Infostop


def detect_stays(pings):
    """Detect consecutive stay episodes from user, latitude, longitude and Unix seconds."""
    episodes = []
    for user, group in pings.groupby('user_id', sort=False):
        points = group.sort_values('timestamp').copy()
        model = Infostop(r1=50, r2=50, min_staying_time=300,
                         max_time_between=86400, verbose=False)
        points['label'] = model.fit_predict(points[['lat', 'lon', 'timestamp']].to_numpy())
        points['episode'] = (
            points.label.ne(points.label.shift()) | points.timestamp.diff().ge(86400)
        ).cumsum()
        stops = points.loc[points.label.ge(0)].groupby('episode').agg(
            lat=('lat', 'median'), lon=('lon', 'median'),
            start=('timestamp', 'min'), end=('timestamp', 'max'),
        )
        stops['duration'] = stops.end - stops.start
        stops['user_id'] = user
        stops['stay_gid9'] = [h3.latlng_to_cell(lat, lon, 9) for lat, lon in zip(stops.lat, stops.lon)]
        episodes.append(stops)
    return pd.concat(episodes, ignore_index=True)


def _select_anchor(stays, window, name):
    """Select each month's most frequent cell and retain its window count."""
    selected = stays.loc[window].copy()
    monthly_keys = ['user_id', 'YY', 'MM']
    counts = selected.groupby([*monthly_keys, 'stay_gid9']).size().rename('stays').reset_index()
    counts['window_stays'] = counts.groupby(monthly_keys).stays.transform('sum')
    counts['probability'] = counts.stays / counts.window_stays
    winners = counts.loc[counts.groupby(monthly_keys).probability.idxmax()]
    return winners.rename(columns={
        'stay_gid9': f'{name}_gid9', 'stays': f'{name}_stays',
        'probability': f'{name}_probability', 'window_stays': f'{name}_window_stays',
    })


def detect_home_work(stays):
    """Use local stay-start times; ties follow sorted H3 IDs."""
    local = pd.to_datetime(stays.start, unit='s', utc=True).dt.tz_convert('Europe/Helsinki')
    observed = stays.assign(YY=local.dt.year, MM=local.dt.month)
    home_window = local.dt.hour.ge(23) | local.dt.hour.lt(7)
    work_window = local.dt.hour.ge(10) & local.dt.hour.lt(16) & local.dt.dayofweek.lt(5)
    homes = _select_anchor(observed, home_window, 'home')
    work = _select_anchor(observed, work_window, 'work')
    keys = ['user_id', 'YY', 'MM']
    months = observed[keys].drop_duplicates()
    return months.merge(homes, on=keys, how='left').merge(work, on=keys, how='left')
