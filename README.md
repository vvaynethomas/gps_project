# GPS trip and visit anomaly exploration

How can the timing and geography of everyday trips help identify unusual visits? This exploratory Python project uses Shanghai GPS trip records to study that question through spatial feature engineering, road-network context, and unsupervised anomaly detection.

The project connects individual **visits** (a location and timestamp) into **trips**, groups trips by **user** and **community**, and experiments with models for distinguishing ordinary observations from synthetic anomalies. It is a research prototype, not a finished tracking application.

## Start exploring

You can read the notebooks on GitHub without downloading the dataset or running any code.

| Entry point | What to explore |
| --- | --- |
| [`final_notebook.ipynb`](final_notebook.ipynb) | Start here for the main experiment: data loading, trip and visit features, synthetic visit anomalies, dimensionality reduction, and model comparisons. |
| [`eda.ipynb`](eda.ipynb) | Earlier spatial and temporal exploration, maps, trip feature engineering, and model experiments. |
| [`package_demo.ipynb`](package_demo.ipynb) | Examples of the module's user/community objects, synthetic trip generation, and detector usage. Some cells load saved Python objects. |
| [`gps_package.py`](gps_package.py) | The reusable pieces: `Visit`, `Trip`, `User`, `Community`, distance/bearing helpers, workbook loading, and `gps_anomaly_detector`. |
| [`tests/`](tests/) | Focused regression tests for import safety, workbook-path configuration, bearings, visit times, and trip retention, using synthetic dependencies. |
| [`app_attempt1.py`](app_attempt1.py) | An unfinished Kivy UI stub, not an analysis entry point. |

## Analysis workflow

The main notebook follows this path:

1. **Load trip records.** Read the Shanghai Excel workbook, skip its six introductory rows, and normalize column names.
2. **Build spatial and temporal features.** Derive calendar/time-of-day fields, trip duration, bearing, great-circle and Manhattan distances, and nearest road-network nodes using OSMnx.
3. **Represent visits.** Separate trip starts and ends into timestamped locations and explore their distribution on the road network.
4. **Create comparison examples.** Generate synthetic visit anomalies and split ordinary visits into training and test sets.
5. **Compare unsupervised models.** Explore scaling and PCA, including two-component and 99%-variance representations, with Isolation Forest, Elliptic Envelope, and One-Class SVM. The Python module also includes Bayesian Gaussian Mixture experiments.

The notebooks contain maps, feature exploration, and model-comparison outputs. Those saved outputs document earlier experiments; they are not independently reproduced benchmarks, and synthetic anomalies do not establish accuracy on real-world anomalous behavior.

## Explore locally

### Environment

Use an isolated Python environment and a Jupyter notebook interface. There is no dependency manifest, pinned environment, or verified Python/package version combination in this repository.

Dependencies visible in the source include:

| Area | Packages |
| --- | --- |
| Core analysis module | NumPy, pandas, Matplotlib, geopy, OSMnx, scikit-learn |
| Notebook exploration | Jupyter, GeoPandas, MovingPandas, Shapely, pyproj, SciPy, seaborn, geoplot, category-encoders |
| Workbook loading | An Excel reader compatible with pandas and `.xlsx` files, such as openpyxl |

Treat this as a dependency starting point, not a tested installation recipe. Some notebook cells use older library APIs and may require adaptation to your environment.

### Data

The analysis uses [Multiday GPS tracking data of Shanghai](https://doi.org/10.21227/fqe8-db87), hosted by IEEE DataPort. Obtain the workbook from that record and consult its current access and reuse terms. The workbook is not included here, and the code does not download it automatically.

From the repository root, set the path before launching Jupyter so the notebook kernel inherits it. For example, in PowerShell, with your environment already activated:

```powershell
$env:GPS_DATA_PATH = 'C:\path\to\shanghai_gps.xlsx'
jupyter notebook
```

Open `final_notebook.ipynb` and inspect the cells before running them. Both analysis notebooks and the module use `GPS_DATA_PATH`; if it is unset, they look for `shanghai_gps.xlsx` in the current working directory. The loader expects the source workbook's layout, including user IDs, start/end timestamps in seconds, and start/end coordinates, rather than an arbitrary GPS export.

For module exploration, loading is explicit:

```python
import gps_package as gps

data, graph = gps.load_location(pickle_graph=False)
```

Importing the module alone does not read the workbook or request a graph. `load_location()` first attempts to load a local graph pickle, then requests a driving graph through OSMnx if that fails. `pickle_graph=False` prevents saving the graph; it does **not** prevent loading an existing pickle or making a network request. Running `gps_package.py` directly also starts a data-dependent model experiment, so it is not a setup check.

### Reproduction boundaries

The notebooks make live OSMnx requests for Shanghai road networks, and some exploration cells use Nominatim geocoding. The main notebook uses a walking graph, while the module defaults to a driving graph, so their spatial context is not identical.

The repository also contains saved graph/detector pickle files. Pickle loading can execute code: inspect the loading cells and use only files you trust. Saved objects can depend on the library versions that created them.

A clean end-to-end notebook or model run has not been verified. The focused tests do not establish workbook/graph compatibility or model quality; browsing the notebooks and source is the simplest way to explore the project without reconstructing the original environment.
