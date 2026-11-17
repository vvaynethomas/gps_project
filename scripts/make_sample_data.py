"""Write a synthetic workbook with the same layout as the Shanghai multiday GPS data.

Each synthetic participant has a home and a work anchor inside the Shanghai bounding box
and alternates activities and trips between them (plus occasional third places) over a
few weeks. The layout matches the source workbook's first sheet: six description rows,
then the header row, then one row per activity or trip.

    python scripts/make_sample_data.py                # data/sample_gps.xlsx, 40 participants
    python scripts/make_sample_data.py --users 10 --days 7 --out /tmp/tiny.xlsx
"""
import argparse
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ['UID', 'Date (In Days)', 'Week', 'Start time (In Seconds)', 'End time (In seconds)',
           'Start Longitude', 'Start Latitude', 'End Longitude', 'End Latitude', 'Activity/Trip',
           'Trip Mode', 'Start location', 'End location', 'Activity type']
DESCRIPTION = [
    'Data description (synthetic sample)',
    'Date: Days from 1970.1.1; Time: Seconds from 1970.1.1 00:00:00 UTC',
    'Activity/Trip: 0-Activity, 1-Trip',
    'Activity type: -1 unknown, 0 home, 1 work, 2 other',
    'Location: -1 Other, 0 Home, 1 Work',
]
LNG, LAT = (121.10, 121.80), (30.90, 31.45)
CST = timezone(timedelta(hours=8))


def jitter(point, rng, scale=0.0004):
    return point[0] + rng.gauss(0, scale), point[1] + rng.gauss(0, scale)


def participant_rows(uid, start_day, days, rng):
    home = (rng.uniform(*LNG), rng.uniform(*LAT))
    work = (rng.uniform(*LNG), rng.uniform(*LAT))
    others = [(rng.uniform(*LNG), rng.uniform(*LAT)) for _ in range(3)]
    mode = rng.choice([1, 2, 3, 4, 5, 7])
    rows = []
    for d in range(days):
        day = start_day + timedelta(days=d)
        workday = day.weekday() < 5
        plan = [(home, 0)]
        if workday:
            plan.append((work, 1))
        if rng.random() < 0.4:
            plan.append((rng.choice(others), -1))
        plan.append((home, 0))
        t = datetime(day.year, day.month, day.day, 7, 0, tzinfo=CST) + timedelta(minutes=rng.randint(0, 120))
        for (here, here_type), (there, there_type) in zip(plan, plan[1:]):
            stay = timedelta(hours=9 if here_type == 1 else rng.uniform(0.5, 3) if here_type == -1 else 0.2)
            a, b = jitter(here, rng), jitter(there, rng)
            rows.append([uid, here_type, here_type, t, t + stay, a, a, 0, 0])
            t += stay
            ride = timedelta(minutes=rng.randint(10, 75))
            rows.append([uid, here_type, there_type, t, t + ride, a, jitter(there, rng), 1, mode])
            t += ride
    out = []
    for uid_, loc_a, loc_b, t0, t1, a, b, kind, trip_mode in rows:
        local = t0.astimezone(CST)
        out.append([uid_, (local.date() - datetime(1970, 1, 1).date()).days, local.isoweekday(),
                    int(t0.timestamp()), int(t1.timestamp()), round(a[0], 8), round(a[1], 8),
                    round(b[0], 8), round(b[1], 8), kind, trip_mode, loc_a, loc_b,
                    loc_a if kind == 0 else 1])
    return out


def make_sample(path, users=40, days=21, seed=7):
    rng = random.Random(seed)
    rows = []
    for i in range(users):
        start = datetime(2014, 3, 3) + timedelta(days=rng.randint(0, 300))
        rows += participant_rows(100 + i, start, rng.randint(max(days // 2, 1), days), rng)
    header = [[line] + [None] * (len(COLUMNS) - 1) for line in DESCRIPTION] + [[None] * len(COLUMNS)]
    sheet = pd.DataFrame(header + [COLUMNS] + rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_excel(path, header=False, index=False, sheet_name='records')
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description='Write a synthetic GPS workbook.')
    parser.add_argument('--out', type=Path, default=REPO_ROOT / 'data' / 'sample_gps.xlsx')
    parser.add_argument('--users', type=int, default=40)
    parser.add_argument('--days', type=int, default=21)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    n = make_sample(args.out, args.users, args.days, args.seed)
    print(f'wrote {n} records for {args.users} synthetic participants to {args.out}')


if __name__ == '__main__':
    main()
