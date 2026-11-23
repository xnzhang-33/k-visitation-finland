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

## Structure

```text
├── notebooks/
│   ├── demos/    # Synthetic worked examples
│   └── figures/  # Figure notebooks
├── src/          # Calculation and plotting modules
├── data/
│   ├── aggregated/  # Figure inputs
│   └── sample/      # Synthetic worked-example inputs
└── output/       # PDF exports, created when the figure notebooks run
```


## Quickstart

Clone the repository and enter its directory:

```bash
git clone https://github.com/xnzhang-33/k-visitation-finland.git
cd k-visitation-finland
```

Create and activate a Python environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate

# Alternatively, with conda
conda create -n k-visitation python=3.11 pip
conda activate k-visitation
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
python -m jupyter lab
```

Finally, open a notebook, select the environment’s kernel and choose Run All. Figure PDFs are saved to output/.

## Content

### 1. K-Visitation framework

The notebooks under `notebooks/demos/` illustrate the framework using a synthetic toy user.

- `00-demo-k-visitation.ipynb` aggregates stays into visits, constructs the two K sets and calculates $q_K$, following the worked example in the Supplementary Information.
- `01-demo-d-epr.ipynb` estimates the user’s exploration rate, simulates visits with d-EPR and calculates $q_K^{null}$.

### 2. Main analysis

The notebooks under `notebooks/figures/` reproduce the main figure panels from the supplied data.

- `01-spatial-alignment.ipynb` — Fig. 1c: the distribution of $q_K$ across Finnish cities.
- `02-travel-time-divergence.ipynb` — Fig. 2: $T_{freq}$, $\Delta T$ and $\Delta T_{rel}$ by $T_{prox}$ group, and bivariate maps of $q_K$ and $\Delta T_{rel}$.
- `03-density-null-model.ipynb` — Fig. 3: empirical and null $q_K$ across cities and $T_{prox}$ groups, together with destination attributes.
- `04-non-alignment-shap.ipynb` — Fig. 4: SHAP summary, feature importance and dependence plots.
- `05-amenity-distance-differentials.ipynb` — Fig. 5 and the supplementary absolute-distance plot: $\Delta d_{rel}$ and $\Delta d$ across 63 amenity classes.

## Data

Access to the raw mobility records and commercial POI data is restricted.  `data/aggregated/` contains aggregated data to reproduce main figures in the study. K-visitation framework is examined from a sample toy user under `data/sample/`. We replaced the original H3 cell identifiers with pseudonymous labels in the data.


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
