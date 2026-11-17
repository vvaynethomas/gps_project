"""End-to-end run of pipeline.py on a small synthetic workbook (needs requirements.txt)."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ('pandas', 'sklearn', 'category_encoders', 'geopy', 'openpyxl', 'matplotlib', 'osmnx')
MISSING = [name for name in REQUIRED if importlib.util.find_spec(name) is None]


@unittest.skipIf(MISSING, f'missing packages: {MISSING}')
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path[:0] = [str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / 'scripts')]
        import make_sample_data
        import pipeline
        cls.pipeline = pipeline
        cls.tmp = tempfile.TemporaryDirectory()
        cls.workbook = Path(cls.tmp.name) / 'sample.xlsx'
        make_sample_data.make_sample(cls.workbook, users=12, days=8, seed=3)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_run_writes_metrics_and_figures(self):
        out = Path(self.tmp.name) / 'outputs'
        summary = self.pipeline.run(self.workbook, n_synthetic=150, output_dir=out)
        self.assertEqual(summary['participants'], 12)
        self.assertEqual(summary['visits'], 2 * summary['records'])
        self.assertEqual(set(summary['models']), {
            'empirical_covariance', 'robust_covariance', 'isolation_forest', 'one_class_svm'})
        for scores in summary['models'].values():
            for value in scores.values():
                self.assertGreaterEqual(value, 0)
                self.assertLessEqual(value, 1)
        self.assertEqual(summary['models']['isolation_forest']['detection_rate_displaced'], 1.0)
        self.assertEqual(json.loads((out / 'metrics.json').read_text()), summary)
        for figure in ('trip_starts_by_hour.png', 'decision_boundaries.png'):
            self.assertTrue((out / 'figures' / figure).stat().st_size > 0)

    def test_road_network_features_use_nearest_node(self):
        import networkx as nx
        import pandas as pd
        graph = nx.MultiDiGraph(crs='epsg:4326')
        graph.add_node(1, x=121.40, y=31.20)
        graph.add_node(2, x=121.50, y=31.30)
        graph.add_edge(1, 2, length=1.0)
        visits = pd.DataFrame({'uid': [1, 1], 'timestamp': [1400000000, 1400003600],
                               'longitude': [121.401, 121.499], 'latitude': [31.201, 31.299]})
        features = self.pipeline.visit_features(visits, graph)
        self.assertEqual(features.nearest_node.tolist(), [1, 2])
        self.assertTrue((features.nn_dist > 0).all())
        self.assertEqual(features.hour_of_day.tolist(), [16, 17])


if __name__ == '__main__':
    unittest.main()
