# GPS trip analysis research prototype

This repository contains an exploratory GPS trip and visit analysis project. It preserves the original notebooks and methods; it is not a maintained or independently reproduced application. The notebooks explore spatial and temporal feature engineering and unsupervised anomaly-detection methods. No performance or operational claims are made here.

## Notebooks

- [`final_notebook.ipynb`](final_notebook.ipynb) is the best starting point for the main experiment. It loads the Shanghai GPS workbook, builds a road graph through OSMnx, engineers trip and visit features, explores synthetic visit anomalies, and compares unsupervised models.
- [`eda.ipynb`](eda.ipynb) contains exploratory spatial and temporal analysis, trip feature engineering, and model experiments. Some cells make live OSMnx or Nominatim requests.
- [`package_demo.ipynb`](package_demo.ipynb) demonstrates the Python module and community/anomaly-detector objects, including cells that load serialized Python objects.

Notebook outputs and execution counts are saved historical artifacts, not evidence that the notebooks run from a clean environment today.

## Code and methods

- [`gps_package.py`](gps_package.py) defines `Visit`, `Trip`, `User`, `Community`, data-loading helpers, and `gps_anomaly_detector`. The work explores time-derived features, bearings, great-circle and Manhattan distances, nearest road-network nodes, and Isolation Forest, Elliptic Envelope, Bayesian Gaussian Mixture, and One-Class SVM models. The notebooks also explore scaling, PCA, and synthetic visit examples.
- [`app_attempt1.py`](app_attempt1.py) is a minimal Kivy application stub, not a documented analysis interface.

These are the methods present in the source, not a claim that a complete analysis or any model result has been reproduced.

## Data and external services

The analysis uses [Multiday GPS tracking data of Shanghai](https://doi.org/10.21227/fqe8-db87), hosted by IEEE DataPort and included in the repository as `shanghai_gps.xlsx`. The notebooks also retrieve Shanghai road-network data through OSMnx, and some cells use Nominatim for geocoding. Consult the dataset record for current access and reuse terms.

`gps_package.load_location()` explicitly loads the workbook and graph. If the local graph pickle cannot be loaded, it may request a graph from OSMnx; by default, it may also save the graph as a pickle. Pickle deserialization can execute code, so only load serialized files from sources you trust.

## Setup and reproducibility

There is no dependency manifest or pinned environment in the repository. Dependencies are inferred from imports in the package and notebooks, and the notebook environment requires additional geospatial and plotting packages. A clean, end-to-end notebook or model run has not been verified.

The package parses and its focused import-safety regression test passes with synthetic dependency stubs. This does not verify workbook or graph loading, notebook execution, or model behavior. Existing saved notebook outputs are not a substitute for a clean run.
