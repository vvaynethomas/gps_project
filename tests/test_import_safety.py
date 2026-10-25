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


if __name__ == "__main__":
    unittest.main()
