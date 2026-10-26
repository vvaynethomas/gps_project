import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_package():
    modules = {}
    for name in (
        "matplotlib", "matplotlib.pyplot", "pandas", "numpy", "geopy",
        "osmnx", "osmnx.distance", "sklearn", "sklearn.model_selection",
        "sklearn.ensemble", "sklearn.decomposition", "sklearn.preprocessing",
        "sklearn.covariance", "sklearn.mixture", "sklearn.svm",
    ):
        modules[name] = types.ModuleType(name)
        modules[name].__path__ = []
    for module_name, names in {
        "sklearn.model_selection": (
            "train_test_split", "cross_val_predict", "cross_val_score", "GridSearchCV"
        ),
        "sklearn.ensemble": ("IsolationForest",),
        "sklearn.decomposition": ("PCA",),
        "sklearn.preprocessing": ("StandardScaler", "normalize"),
        "sklearn.covariance": ("EllipticEnvelope",),
        "sklearn.mixture": ("BayesianGaussianMixture",),
        "sklearn.svm": ("OneClassSVM",),
    }.items():
        for name in names:
            setattr(modules[module_name], name, object)
    modules["pandas"].DataFrame = type("DataFrame", (), {})
    modules["pandas"].Series = type("Series", (), {})
    modules["numpy"].min = min
    modules["geopy"].distance = types.SimpleNamespace(
        great_circle=lambda *args: types.SimpleNamespace(km=1)
    )
    spec = importlib.util.spec_from_file_location(
        "gps_package", REPOSITORY_ROOT / "gps_package.py"
    )
    package = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(package)
    return package


class GPSBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gps = load_package()

    def make_trip(self):
        return self.gps.Trip(full=[
            self.gps.Visit(uid=7, timestamp=60, latitude=0, longitude=170),
            self.gps.Visit(uid=7, timestamp=120, latitude=0, longitude=-170),
        ])

    def test_bearing_uses_degree_coordinates_and_returns_radians(self):
        for coordinates, expected in (
            ((0, 0, 45, 0), 0),
            ((0, 0, 0, 90), math.pi / 2),
            ((0, 0, -45, 0), math.pi),
            ((0, 0, 0, -90), 3 * math.pi / 2),
            ((0, 170, 0, -170), math.pi / 2),
            ((0, 0, 45, 90), math.pi / 4),
            ((45, 0, 0, 90), math.pi / 2),
        ):
            with self.subTest(coordinates=coordinates):
                bearing = self.gps.calculate_bearing(*coordinates)
                self.assertAlmostEqual(bearing, expected, places=6)
                self.assertGreaterEqual(bearing, 0)
                self.assertLess(bearing, 2 * math.pi)

    def test_trip_uses_correct_bearing_without_changing_coordinates(self):
        trip = self.make_trip()
        self.assertAlmostEqual(trip.bearing, math.pi / 2)
        self.assertEqual(trip.start_loc, (0, 170))
        self.assertEqual(trip.end_loc, (0, -170))
        self.assertEqual(trip.as_row()["bearing"], trip.bearing)

    def test_user_retains_trips_visits_and_locations_when_visits_omitted(self):
        trip = self.make_trip()
        user = self.gps.User(id=7, trips=[trip])
        self.assertEqual(user.trips, [trip])
        self.assertEqual(user.visits, trip.visits)
        self.assertEqual(user.locations, [trip.start_loc, trip.end_loc])
        self.assertEqual(self.gps.Community(users=[user]).trips, [trip])

    def test_user_retains_trips_when_visits_empty(self):
        trip = self.make_trip()
        user = self.gps.User(id=7, trips=[trip], visits=[])
        self.assertEqual(user.trips, [trip])
        self.assertEqual(user.visits, trip.visits)
        self.assertEqual(user.locations, [trip.start_loc, trip.end_loc])

    def test_user_adds_explicit_visits_once_alongside_trip_visits(self):
        trip = self.make_trip()
        visit = self.gps.Visit(uid=7, timestamp=180, latitude=1, longitude=2)
        user = self.gps.User(id=7, trips=[trip], visits=[visit])
        self.assertEqual(user.trips, [trip])
        self.assertEqual(user.visits, trip.visits + [visit])
        self.assertEqual(user.locations, [trip.start_loc, trip.end_loc, visit.location])

    def test_user_without_data_has_empty_collections(self):
        user = self.gps.User(id=7)
        self.assertEqual(user.trips, [])
        self.assertEqual(user.visits, [])
        self.assertEqual(user.locations, [])
        self.assertEqual(user.connections, [])

    def test_visit_time_transformation_copies_frame_and_supports_inplace(self):
        for inplace in (False, True):
            with self.subTest(inplace=inplace):
                community = self.gps.Community(users=[self.gps.User(id=7)])
                timestamps = [60, 120]
                columns = {"timestamp": timestamps}
                transformed = MagicMock()
                transformed.__getitem__.side_effect = columns.__getitem__
                transformed.__setitem__.side_effect = columns.__setitem__
                source = Mock()
                source.copy.return_value = transformed
                community.get_visits_df = Mock(
                    spec=community.get_visits_df, return_value=source
                )
                fields = dict(
                    month=[1, 1], day=[1, 1], dayofweek=[3, 3],
                    hour=[0, 0], minute=[1, 2],
                )
                datetimes = types.SimpleNamespace(dt=types.SimpleNamespace(**fields))
                with patch.object(
                    self.gps.pd, "to_datetime", return_value=datetimes, create=True
                ) as to_datetime:
                    result = community.transform_time(data_type="visits", inplace=inplace)
                community.get_visits_df.assert_called_once_with()
                source.copy.assert_called_once_with()
                to_datetime.assert_called_once_with(timestamps, unit="s")
                self.assertIs(result, transformed)
                for column, field in (
                    ("month", "month"), ("day", "day"), ("day_of_week", "dayofweek"),
                    ("hour_of_day", "hour"), ("minute_of_hour", "minute"),
                ):
                    self.assertEqual(columns[column], fields[field])
                if inplace:
                    transformed.drop.assert_called_once_with(["timestamp"], axis=1, inplace=True)
                    self.assertIs(community.train, transformed)
                else:
                    transformed.drop.assert_not_called()
                    self.assertFalse(hasattr(community, "train"))


if __name__ == "__main__":
    unittest.main()
