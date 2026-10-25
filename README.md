# GPS trip analysis research prototype

This repository contains an exploratory GPS trip and visit analysis project. It preserves the original notebooks and methods; it is not a maintained or independently reproduced application. The notebooks explore spatial and temporal feature engineering and unsupervised anomaly-detection methods. No performance or operational claims are made here.

## Notebooks

- [`final_notebook.ipynb`](final_notebook.ipynb) is the best starting point for the main experiment. It loads the Shanghai GPS workbook, builds a road graph through OSMnx, engineers trip and visit features, explores synthetic visit anomalies, and compares unsupervised models.
- [`eda.ipynb`](eda.ipynb) contains exploratory spatial and temporal analysis, trip feature engineering, and model experiments. Some cells make live OSMnx or Nominatim requests.
- [`package_demo.ipynb`](package_demo.ipynb) demonstrates the Python module and community/anomaly-detector objects. It also contains cells that deserialize `full_comm_ad.p`; do not run those cells unless you have verified that the pickle came from a trusted source.

Notebook outputs and execution counts are saved historical artifacts, not evidence that the notebooks run from a clean environment today.

## Code and methods

- [`gps_package.py`](gps_package.py) defines `Visit`, `Trip`, `User`, `Community`, data-loading helpers, and `gps_anomaly_detector`. The work explores time-derived features, bearings, great-circle and Manhattan distances, nearest road-network nodes, and Isolation Forest, Elliptic Envelope, Bayesian Gaussian Mixture, and One-Class SVM models. The notebooks also explore scaling, PCA, and synthetic visit examples.
- [`app_attempt1.py`](app_attempt1.py) is a minimal Kivy application stub, not a documented analysis interface.

These are the methods present in the source, not a claim that a complete analysis or any model result has been reproduced.

## Data provenance and privacy

The repository contains `shanghai_gps.xlsx` and serialized graph/model artifacts, and the notebooks load the workbook as an Excel file. The repository does not provide its source citation, collection details, consent information, license, or redistribution terms. Those rights and provenance have not been verified.

GPS traces can reveal sensitive movements and locations. Do not publish, redistribute, or add trajectory examples from these files until the maintainer has verified the data source, privacy constraints, and permission to share them. The data and research artifacts are retained unchanged pending that review; this documentation adds no trajectory samples.

`gps_package.load_location()` is now an explicit initialization step rather than an import side effect. Calling it reads the workbook, attempts to deserialize the configured graph pickle, and may request a graph from OSMnx if that load fails; with its default options it can also write a pickle. Only call it when the local files are trusted and any required network access is acceptable. The package demo's pickle-loading cells have the same trust requirement.

## Setup and reproduction status

There is no dependency manifest or pinned environment in the repository. Dependencies are inferred from imports in the package and notebooks, and the notebook environment requires additional geospatial and plotting packages. A clean, end-to-end notebook or model run has not been verified.

The package previously failed to parse because of an incomplete duplicate `generate_visit` definition and loaded its workbook and graph during import. [PR #3](https://github.com/vvaynethomas/gps_project/pull/3) removed the malformed, unused package definition and made data/graph loading explicit. Its focused regression test checks parsing and import safety with synthetic dependency stubs; it does not validate data loading, graph retrieval, notebook execution, or model behavior. The current Python environment used for that check did not have the project's third-party dependencies installed.

Before attempting reproduction, verify the data provenance and sharing rights, establish a supported dependency environment, and use an approved non-sensitive sample if one is available. Treat existing saved notebook outputs as historical until a clean run has been validated.
