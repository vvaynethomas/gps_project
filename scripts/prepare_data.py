"""Check a downloaded copy of the Shanghai multiday GPS workbook and put it in data/.

IEEE DataPort serves the dataset to signed-in users, so download the workbook from
https://doi.org/10.21227/fqe8-db87 in a browser first, then:

    python scripts/prepare_data.py ~/Downloads/<downloaded workbook>.xlsx

The script confirms the first sheet has the expected header row, reports record and
participant counts, compares the file's SHA-256 with the version this project was built
on, and copies it to data/shanghai_multiday_gps.xlsx (the default path for pipeline.py,
gps_package.py and the notebooks).
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / 'data' / 'shanghai_multiday_gps.xlsx'
KNOWN_SHA256 = 'aaf900db4e450790c316698733997c7b9a1b86c715205df2ade151f2afd4080f'
EXPECTED_HEADER = ['UID', 'Date (In Days)', 'Week', 'Start time (In Seconds)', 'End time (In seconds)',
                   'Start Longitude', 'Start Latitude', 'End Longitude', 'End Latitude', 'Activity/Trip',
                   'Trip Mode', 'Start location', 'End location', 'Activity type']


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Validate and install the GPS workbook.')
    parser.add_argument('source', type=Path, help='the workbook downloaded from IEEE DataPort')
    parser.add_argument('--target', type=Path, default=TARGET)
    args = parser.parse_args()

    records = pd.read_excel(args.source, sheet_name=0, skiprows=6, header=0)
    header = [str(c).strip() for c in records.columns]
    if header[:len(EXPECTED_HEADER)] != EXPECTED_HEADER:
        sys.exit(f'unexpected header in the first sheet: {header}')
    print(f'{len(records):,} records from {records.UID.nunique():,} participants')

    digest = sha256(args.source)
    if digest == KNOWN_SHA256:
        print('matches the version used for the published results')
    else:
        print(f'sha256 {digest} differs from the version used here ({KNOWN_SHA256[:12]}...); '
              'results will differ if the records changed')

    args.target.parent.mkdir(parents=True, exist_ok=True)
    if args.source.resolve() != args.target.resolve():
        shutil.copyfile(args.source, args.target)
    print(f'installed at {args.target}')


if __name__ == '__main__':
    main()
