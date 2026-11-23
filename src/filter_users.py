"""User selection from complete stay and home/work results."""

from detect_stays import detect_home_work


def filter_users(stays, min_window_stays):
    """Select users by window counts and home/work availability."""
    anchors = detect_home_work(stays)
    counts = anchors.groupby('user_id')[['home_window_stays', 'work_window_stays']].sum()
    sufficient = counts.index[counts.ge(min_window_stays).all(axis=1)]
    complete = anchors.dropna(subset=['home_gid9', 'work_gid9']).user_id.unique()
    eligible = sufficient.intersection(complete)
    return (stays.loc[stays.user_id.isin(eligible)].copy(),
            anchors.loc[anchors.user_id.isin(eligible)].copy())
