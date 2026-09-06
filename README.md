# Recurrent visitations reveal selectivity beyond the 15-Minute City vision

Repository accompanying the manuscript *Recurrent visitations reveal selectivity beyond the 15-Minute City vision*. Code and aggregated data to reproduce its main figure panels, plus self-contained demos of the K-Visitation and d-EPR framework on synthetic data. Raw mobility data and model fitting are not included.

## Authors

- Xiuning Zhang¹
- Alexei Poliakov²
- Henrikki Tenkanen³
- Elsa Arcaute¹

¹ The Centre for Advanced Spatial Analysis, University College London, London, UK<br>
² Locomizer Ltd, London, UK<br>
³ Department of Built Environment, Aalto University, Espoo, Finland

Pre-print available on [arXiv](https://doi.org/10.48550/arXiv.2509.00919).

## Quickstart

Figure reproduction notebooks under [`notebooks/`](notebooks/) (read from `data/`, write to `output/`):

- [`notebooks/01-spatial-alignment.ipynb`](notebooks/01-spatial-alignment.ipynb) — Figure 1c
- [`notebooks/02-travel-time-divergence.ipynb`](notebooks/02-travel-time-divergence.ipynb) — Figure 2
- [`notebooks/03-density-null-model.ipynb`](notebooks/03-density-null-model.ipynb) — Figure 3
- [`notebooks/04-non-alignment-shap.ipynb`](notebooks/04-non-alignment-shap.ipynb) — Figure 4
- [`notebooks/05-amenity-distance-differentials.ipynb`](notebooks/05-amenity-distance-differentials.ipynb) — Figure 5

Framework demos (synthetic data):

- [`notebooks/demo-k_visitation.ipynb`](notebooks/demo-k_visitation.ipynb) — K-visitation demo
- [`notebooks/demo-depr.ipynb`](notebooks/demo-depr.ipynb) — d-EPR null-model demo

## Requirements

- Python 3.10+
- Install with `pip install -r requirements.txt`
- Run notebooks with `jupyter lab` or via `nbconvert`

## Data access

The raw mobility data cannot be shared because of privacy restrictions. Aggregated data needed to reproduce the figures is under [`data/`](data/). [`sample_data_k_visitation.csv`](data/sample_data_k_visitation.csv) is synthetic.

## Structure

```text
├── notebooks/
│   ├── 01-spatial-alignment.ipynb               # Figure 1c
│   ├── 02-travel-time-divergence.ipynb          # Figure 2
│   ├── 03-density-null-model.ipynb              # Figure 3
│   ├── 04-non-alignment-shap.ipynb              # Figure 4
│   ├── 05-amenity-distance-differentials.ipynb  # Figure 5
│   ├── demo-k_visitation.ipynb                  # K-visitation demo
│   └── demo-depr.ipynb                          # d-EPR null-model demo
├── src/                          # Reusable framework modules and style
│   ├── figure_style.py
│   ├── k_visitation.py
│   ├── mobility_utils.py
│   └── distance_differentials.py
├── data/                         # Aggregated data for figure reproduction
│   ├── sample_data_k_visitation.csv
│   └── ...
└── requirements.txt
```

## Citation

Please cite the manuscript as:

```bibtex
@misc{zhang2026recurrent,
  title   = {Recurrent Visitations Reveal Selectivity beyond the 15-Minute City Vision},
  author  = {Zhang, Xiuning and Poliakov, Alexei and Tenkanen, Henrikki and Arcaute, Elsa},
  year    = {2026},
  eprint  = {2509.00919},
  archiveprefix = {arXiv},
  doi     = {10.48550/arXiv.2509.00919},
  url     = {https://arxiv.org/abs/2509.00919}
}
```
