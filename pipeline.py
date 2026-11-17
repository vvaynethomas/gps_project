"""End-to-end visit anomaly experiment on the Shanghai multiday GPS workbook.

Reproduces the experiment in final_notebook.ipynb as a script:

1. load trip/activity records from the workbook's first sheet
2. engineer trip features (duration, calendar fields, distances, bearing)
3. split every record into two timestamped visits (start and end)
4. generate synthetic anomalous visits
5. hash-encode identifiers, scale, reduce with PCA (99% variance)
6. fit Elliptic Envelope (empirical and robust), Isolation Forest and One-Class SVM
   on ordinary visits and score them on held-out visits and the synthetic anomalies

Usage (from the repository root):

    python pipeline.py                                  # data/shanghai_multiday_gps.xlsx
    python pipeline.py --data data/sample_gps.xlsx      # synthetic sample
    python pipeline.py --road-network                   # add nearest walking-network node features

Writes outputs/metrics.json and figures in outputs/figures/.
"""
import argparse
import json
import random
import string
from pathlib import Path

import numpy as np
import pandas as pd
from geopy import distance as geodist
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import OneClassSVM

from gps_package import DATA_PATH, OUTPUT_DIR, calculate_bearing

# Prior used in the notebook: kidnapping prevalence (2.8 per 1,000) times the share of
# abductions that happen away from home (7/10), spread over a year of days.
ANOMALY_FRACTION = (2.8 / 1000) * (7 / 10) / 365
ONE_YEAR_SECONDS = 365 * 24 * 3600
FEATURE_DROP = ['date_in_days', 'week', 'activitytrip', 'trip_mode',
                'start_location', 'end_location', 'activity_type']


TRIP_MODES = {1: 'walk', 2: 'bicycle', 3: 'e-bicycle', 4: 'car', 5: 'bus', 6: 'shuttle', 7: 'subway', 8: 'other'}


def load_records(path):
    """Read the trip/activity sheet (six description rows above the header)."""
    data = pd.read_excel(path, sheet_name=0, skiprows=6, header=0)
    data.columns = ['_'.join(name.lower().translate(str.maketrans(' ', ' ', string.punctuation)).split())
                    for name in data.columns]
    return data


def load_trips(path):
    """Records reduced to the columns the models use: uid, start/end time and coordinates."""
    data = load_records(path)
    return data.drop(columns=[c for c in FEATURE_DROP if c in data.columns])


def travel_summary(records):
    """Descriptive statistics over the rows coded as trips (Activity/Trip == 1)."""
    trips = trip_features(records[records.activitytrip == 1].drop(columns=FEATURE_DROP))
    modes = records.loc[records.activitytrip == 1, 'trip_mode'].map(TRIP_MODES).value_counts(normalize=True)
    return {
        'trips': int(len(trips)),
        'activities': int((records.activitytrip == 0).sum()),
        'records_per_participant_median': float(records.groupby('uid').size().median()),
        'trip_duration_min_median': round(float(trips.duration_in_seconds.median() / 60), 1),
        'trip_distance_km_median': round(float(trips.gc_dist.median()), 2),
        'trip_distance_km_p90': round(float(trips.gc_dist.quantile(0.9)), 2),
        'mode_share': {k: round(float(v), 3) for k, v in modes.items()},
    }


def add_time_fields(frame, column, prefix):
    stamp = pd.to_datetime(frame[column], unit='s')
    frame[f'{prefix}month'] = stamp.dt.month
    frame[f'{prefix}day_of_month'] = stamp.dt.day
    frame[f'{prefix}day_of_week'] = stamp.dt.dayofweek
    frame[f'{prefix}hour_of_day'] = stamp.dt.hour
    frame[f'{prefix}minute_of_hour'] = stamp.dt.minute
    return frame


def trip_features(trips):
    """Duration, calendar fields, great-circle/Manhattan distance and bearing per trip."""
    X = trips.copy()
    X['trip_id'] = X.uid.astype(int).astype(str) + X.start_time_in_seconds.astype(int).astype(str)
    X['duration_in_seconds'] = X.end_time_in_seconds - X.start_time_in_seconds
    X = add_time_fields(X, 'start_time_in_seconds', 'start_')
    X = add_time_fields(X, 'end_time_in_seconds', 'end_')
    coords = X[['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude']].to_numpy()
    X['gc_dist'] = [geodist.great_circle((a, b), (c, d)).km for a, b, c, d in coords]
    X['mn_dist'] = [geodist.great_circle((a, b), (a, d)).km + geodist.great_circle((a, b), (c, b)).km
                    for a, b, c, d in coords]
    X['bearing'] = [calculate_bearing(a, b, c, d) for a, b, c, d in coords]
    return X


def visits_from_trips(trips):
    """Every record contributes its start and its end as a timestamped visit."""
    columns = ['uid', 'timestamp', 'longitude', 'latitude']
    start = trips[['uid', 'start_time_in_seconds', 'start_longitude', 'start_latitude']].set_axis(columns, axis=1)
    end = trips[['uid', 'end_time_in_seconds', 'end_longitude', 'end_latitude']].set_axis(columns, axis=1)
    return pd.concat([start, end], ignore_index=True)


def visit_features(visits, graph=None):
    X = add_time_fields(visits.copy(), 'timestamp', '')
    if graph is not None:
        import osmnx as ox
        X['nearest_node'], X['nn_dist'] = ox.distance.nearest_nodes(
            graph, X.longitude.to_numpy(), X.latitude.to_numpy(), return_dist=True)
    return X


def synthetic_visits(visits, n, rng):
    """Generate `n` anomalous visits, alternating between two kinds.

    displaced: a participant at a random time and a location outside the study area's
               bounding box (anywhere on the globe beyond it)
    cross:     a participant at a random time at a place another participant visited
    """
    lat_min, lat_max = visits.latitude.min(), visits.latitude.max()
    lng_min, lng_max = visits.longitude.min(), visits.longitude.max()
    uids = visits.uid.unique()
    t_min, t_max = int(visits.timestamp.min()), int(visits.timestamp.max()) + ONE_YEAR_SECONDS
    rows = []
    for i in range(n):
        uid = rng.choice(uids)
        kind = 'cross' if i % 3 == 0 else 'displaced'
        timestamp = rng.randint(t_min, t_max)
        if kind == 'cross':
            other = visits[visits.uid != uid].sample(1, random_state=rng.randint(0, 2 ** 31)).iloc[0]
            lat, lng = other.latitude, other.longitude
        else:
            lat = rng.choice([rng.uniform(lat_max, 90), rng.uniform(-90, lat_min)])
            lng = rng.choice([rng.uniform(lng_max, 180), rng.uniform(-180, lng_min)])
        rows.append({'uid': uid, 'timestamp': timestamp, 'longitude': lng, 'latitude': lat, 'kind': kind})
    return pd.DataFrame(rows)


def build_models(fraction):
    return {
        'empirical_covariance': EllipticEnvelope(support_fraction=1., contamination=fraction),
        'robust_covariance': EllipticEnvelope(contamination=fraction),
        'isolation_forest': IsolationForest(contamination=fraction, random_state=13),
        'one_class_svm': OneClassSVM(nu=fraction, gamma='auto'),
    }


def run(data_path, road_network=False, place='Shanghai, China', n_synthetic=2000, seed=13,
        output_dir=OUTPUT_DIR, figures=True):
    import category_encoders as ce

    rng = random.Random(seed)
    np.random.seed(seed)
    output_dir = Path(output_dir)
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)

    records = load_records(data_path)
    trips = trip_features(records.drop(columns=[c for c in FEATURE_DROP if c in records.columns]))
    graph = None
    if road_network:
        import osmnx as ox
        graph = ox.graph_from_place(place, network_type='walk', simplify=True, retain_all=False,
                                    truncate_by_edge=True)
    visits = visit_features(visits_from_trips(trips), graph)
    synthetic = synthetic_visits(visits, n_synthetic, rng)
    kinds = synthetic.pop('kind').to_numpy()
    synthetic = visit_features(synthetic, graph)

    hash_cols = ['uid'] + (['nearest_node'] if graph is not None else [])
    encoder = ce.HashingEncoder(cols=hash_cols).fit(visits)
    normal = encoder.transform(visits)
    abnormal = encoder.transform(synthetic[visits.columns])
    X_train, X_test = train_test_split(normal, random_state=seed)

    reducer = Pipeline([('scaler', StandardScaler()), ('reduce_dim', PCA(n_components=0.99))]).fit(X_train)
    train_pc, test_pc, out_pc = (reducer.transform(x) for x in (X_train, X_test, abnormal))

    results = {}
    for name, model in build_models(ANOMALY_FRACTION).items():
        model.fit(train_pc)
        flagged_train = model.predict(train_pc) == -1
        flagged_test = model.predict(test_pc) == -1
        flagged_out = model.predict(out_pc) == -1
        results[name] = {
            'false_positive_rate_train': round(float(flagged_train.mean()), 5),
            'false_positive_rate_test': round(float(flagged_test.mean()), 5),
            'detection_rate_synthetic': round(float(flagged_out.mean()), 5),
            **{f'detection_rate_{k}': round(float(flagged_out[kinds == k].mean()), 5)
               for k in sorted(set(kinds))},
        }

    summary = {
        'records': int(len(trips)),
        'participants': int(trips.uid.nunique()),
        'visits': int(len(visits)),
        'train_visits': int(len(X_train)),
        'test_visits': int(len(X_test)),
        'synthetic_visits': {k: int((kinds == k).sum()) for k in sorted(set(kinds))},
        'features_before_pca': int(X_train.shape[1]),
        'pca_components_99pct': int(train_pc.shape[1]),
        'road_network_features': graph is not None,
        'anomaly_fraction': ANOMALY_FRACTION,
        'travel': travel_summary(records),
        'models': results,
    }
    (output_dir / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    if figures:
        plot_trip_starts(records, output_dir / 'figures' / 'trip_starts_by_hour.png')
        plot_decision_boundaries(X_train, abnormal, output_dir / 'figures' / 'decision_boundaries.png')
    return summary


def plot_trip_starts(records, path):
    """Share of trips starting in each local hour, weekdays against weekends."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    starts = pd.to_datetime(records.loc[records.activitytrip == 1, 'start_time_in_seconds'], unit='s')
    starts = starts + pd.Timedelta(hours=8)
    weekend = starts.dt.dayofweek >= 5
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, mask in (('weekday', ~weekend), ('weekend', weekend)):
        share = starts[mask].dt.hour.value_counts(normalize=True).reindex(range(24), fill_value=0)
        ax.plot(share.index, share.values * 100, marker='o', label=f'{label} ({int(mask.sum()):,} trips)')
    ax.set_xlabel('hour of day (UTC+8)')
    ax.set_ylabel('% of trips starting')
    ax.set_xticks(range(0, 24, 3))
    ax.legend(frameon=False)
    ax.set_title('Trip start times')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_decision_boundaries(X_train, X_outliers, path):
    """Fit each detector on two principal components and draw its decision boundary."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    reducer = Pipeline([('scaler', StandardScaler()), ('normalizer', Normalizer()),
                        ('reduce_dim', PCA(n_components=2))]).fit(X_train)
    train_2d, out_2d = reducer.transform(X_train), reducer.transform(X_outliers)
    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 300), np.linspace(-2, 2, 300))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(train_2d[:, 0], train_2d[:, 1], s=2, c='0.6', label='ordinary visits')
    ax.scatter(out_2d[:, 0], out_2d[:, 1], s=4, c='tab:red', alpha=0.5, label='synthetic anomalies')
    handles = []
    for (name, model), color in zip(build_models(ANOMALY_FRACTION).items(), ['m', 'g', 'b', 'k']):
        model.fit(train_2d)
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors=color)
        handles.append(plt.Line2D([], [], color=color, lw=2, label=name.replace('_', ' ')))
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('P1')
    ax.set_ylabel('P2')
    ax.legend(handles=ax.get_legend_handles_labels()[0] + handles, loc='upper right', frameon=False)
    ax.set_title('Detector boundaries on two principal components')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--data', type=Path, default=DATA_PATH, help='path to the GPS workbook')
    parser.add_argument('--road-network', action='store_true',
                        help='download the walking network with OSMnx and add nearest-node features')
    parser.add_argument('--place', default='Shanghai, China')
    parser.add_argument('--synthetic', type=int, default=2000, help='number of synthetic anomalies')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    if not args.data.is_file():
        parser.error(f'{args.data} not found; see README "Get the data" or run scripts/make_sample_data.py')
    summary = run(args.data, road_network=args.road_network, place=args.place,
                  n_synthetic=args.synthetic, seed=args.seed, output_dir=args.output_dir)
    print(json.dumps(summary['models'], indent=2))
    print(f'wrote {args.output_dir / "metrics.json"}')


if __name__ == '__main__':
    main()
