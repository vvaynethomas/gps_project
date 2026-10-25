import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ImportSafetyTests(unittest.TestCase):
    def test_import_does_not_load_data_or_graph(self):
        script = r"""
import sys
import types

def stub_module(name, package=False):
    module = types.ModuleType(name)
    if package:
        module.__path__ = []
    sys.modules[name] = module
    return module

matplotlib = stub_module("matplotlib", package=True)
stub_module("matplotlib.pyplot")
pandas = stub_module("pandas")
numpy = stub_module("numpy")
geopy = stub_module("geopy", package=True)
geopy.distance = types.SimpleNamespace()
osmnx = stub_module("osmnx", package=True)
osmnx.distance = types.SimpleNamespace()

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
    module = stub_module(module_name, package=True)
    for name in names:
        setattr(module, name, object)
    parent_name = module_name.rpartition(".")[0]
    parent = sys.modules.setdefault(parent_name, types.ModuleType(parent_name))
    parent.__path__ = []

sys.path.insert(0, sys.argv[1])
import gps_package
assert gps_package.full_data is None
assert gps_package.G is None
"""
        with tempfile.TemporaryDirectory() as empty_directory:
            result = subprocess.run(
                [sys.executable, "-c", script, str(REPOSITORY_ROOT)],
                cwd=empty_directory,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")


class DataLoadingTests(unittest.TestCase):
    def test_load_location_uses_configured_path_and_missing_data_fails_early(self):
        script = r"""
import os
import sys
import types

def stub_module(name, package=False):
    module = types.ModuleType(name)
    if package:
        module.__path__ = []
    sys.modules[name] = module
    return module

stub_module("matplotlib", package=True)
stub_module("matplotlib.pyplot")
pandas = stub_module("pandas")
stub_module("numpy")
geopy = stub_module("geopy", package=True)
geopy.distance = types.SimpleNamespace()
osmnx = stub_module("osmnx", package=True)
osmnx.distance = types.SimpleNamespace()
graph_requests = []
osmnx.graph_from_place = lambda *args, **kwargs: graph_requests.append(args) or object()

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
    module = stub_module(module_name, package=True)
    for name in names:
        setattr(module, name, object)
    parent_name = module_name.rpartition(".")[0]
    parent = sys.modules.setdefault(parent_name, types.ModuleType(parent_name))
    parent.__path__ = []

class FakeData:
    columns = ["Start Time"]

loaded_paths = []
pandas.read_excel = lambda path, **kwargs: loaded_paths.append((path, kwargs)) or FakeData()
sys.path.insert(0, sys.argv[1])
import gps_package

os.environ["GPS_DATA_PATH"] = sys.argv[2]
data, graph = gps_package.load_location(pickle_graph=False)
assert loaded_paths == [(sys.argv[2], {"skiprows": 6, "header": 0})]
assert data.columns == ["start_time"]
assert len(graph_requests) == 1

os.environ["GPS_DATA_PATH"] = sys.argv[3]
try:
    gps_package.load_location(pickle_graph=False)
except FileNotFoundError as error:
    assert "https://doi.org/10.21227/fqe8-db87" in str(error)
else:
    raise AssertionError("missing workbook should fail")
assert len(graph_requests) == 1
assert len(loaded_paths) == 1
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            workbook_path = Path(temporary_directory) / "synthetic.xlsx"
            workbook_path.touch()
            missing_path = Path(temporary_directory) / "missing.xlsx"
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(REPOSITORY_ROOT),
                    str(workbook_path),
                    str(missing_path),
                ],
                cwd=temporary_directory,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
