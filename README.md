## Recurrent visitations expose the paradox of human mobility in the 15-Minute City vision

This is a repository to accompany 'Recurrent visitations expose the paradox of human mobility in the 15-Minute City vision'. It complements the results presented in the paper with code to reproduce the main results used for this paper.

### Authors

* Xiuning Zhang <sup>1</sup> [<img src="https://i.vimeocdn.com/portrait/4202369_60x60?sig=9bca2cf9bcca8e574a01728a2766c9402e9679900c285517717017ebcae9e227&v=1" alt="ORCID" height="16">](https://orcid.org/0009-0001-6255-4426)
* Alexei Poliakov <sup>2</sup> [<img src="https://i.vimeocdn.com/portrait/4202369_60x60?sig=9bca2cf9bcca8e574a01728a2766c9402e9679900c285517717017ebcae9e227&v=1" alt="ORCID" height="16">](https://orcid.org/0000-0001-7428-4040)
* Henrikki Tenkanen <sup>3</sup> [<img src="https://i.vimeocdn.com/portrait/4202369_60x60?sig=9bca2cf9bcca8e574a01728a2766c9402e9679900c285517717017ebcae9e227&v=1" alt="ORCID" height="16">](https://orcid.org/0000-0002-0918-4710)
* Elsa Arcaute <sup>1</sup> [<img src="https://i.vimeocdn.com/portrait/4202369_60x60?sig=9bca2cf9bcca8e574a01728a2766c9402e9679900c285517717017ebcae9e227&v=1" alt="ORCID" height="16">](https://orcid.org/0000-0001-6579-3265)

Affiliations:<br>
<sup>1</sup> The Centre for Advanced Spatial Analysis, University College London, London, UK <br>
<sup>2</sup> Locomizer Ltd, London, UK <br>
<sup>3</sup> Department of Built Environment, Aalto University, Espoo, Finland <br>

----
Pre-print available [on arXiv](https://arxiv.org/abs/2509.00919). 

----

## Quickstart

0. `notebooks/demo-k_visit.ipynb` → Main framework of the study
1. `notebooks/001-qk.ipynb` → Figures 2 & 3
2. `notebooks/002-travel_time.ipynb` → Figure 1
3. `notebooks/003-xgboost_classifier.ipynb` → Figure 4
4. `notebooks/004-amenity.ipynb` → Figure 5
5. `notebooks/005-elas_seg.ipynb` → Figure 6
6. `notebooks/demo-k_visit.ipynb` → Sample of d-EPR model

## Requirements

- Python 3.10+ recommended
- Install dependencies: `pip install -r requirements.txt`
- Run notebooks with: `jupyter lab` (or `jupyter notebook`)

## Data Access

- Data availability: Due to privacy concerns, the raw mobility data cannot be shared. We have provided aggregated data for reproducibility of the study.
- Place data under:
  - `data/` ...

## Structure 

```
├── src/
│   ├── k_visitation.py           # K-visitation algorithm ($K_{freq}$, $K_{dist}$ and $q_K$)
│   ├── mobility_utils.py         # d-EPR framework for synthesised mobility
│   ├── distance_differentials.py # Calculate distance differentials across amenity categories
│   └── segregation_elasticity.py # Elasticity of segregation
├── notebooks/
│   ├── demo-k_visit.ipynb        # Operationalised demo for K-visitation with anonymised sample data
│   ├── demo-depr.ipynb           # Operationalised demo for d-EPR null model with randomised data
│   ├── 001-qk.ipynb              # Alignment coefficient ($q_K$) analysis (Figs 2 & 3)
│   ├── 002-travel_time.ipynb     # Travel time analysis (Fig 1)
│   ├── 003-xgboost_classifier.ipynb # Non-proximate travel prediction (Fig 4)
│   ├── 004-amenity.ipynb         # Amenity hierarchy (Fig 5)
│   └── 005-elas_seg.ipynb        # Segregation elasticity analysis (Fig 6)
└── data/                         
    ├── ...                       # Data used to reproduce figures in the study
```

## Abstract

In the transition towards sustainability and equity, proximity-centred planning has been adopted in cities worldwide. Exemplified by the 15-Minute City (15mC), this emerging planning paradigm assumes that proximate amenity provision translates into localised utilisation, yet evidence on actual mobility behaviour remains limited. We advance a behaviourally grounded assessment by introducing the *K-Visitation* framework, which identifies the minimal set of distinct visitations needed to cover essential amenities under two orderings: one based on observed visitation frequency ($K_{freq}$), and the other based on proximity to home ($K_{dist}$). Applying it to an 18-month, anonymised mobility data from Finland containing 720 thousand users, we directly compared local mobility potentials with recurrent destination choices, revealing a paradox of human mobility within the 15mC framework. A clear misalignment is observed between proximity and recurrent behaviour, most pronounced in urban cores–areas boast with amenities and traditionally viewed as ideal settings for local living–where residents voluntarily overshoot nearest options, while peripheral routines remain more locally constrained. The paradox further revealed asymmetric functions influences, as compared with everyday amenities, individual travels significantly further for to encounter specialised functions. Furthermore, the social consequences of localism are spatially contingent: increased reliance on local options reduces experienced segregation in central districts but can exacerbate it elsewhere. Our findings stress that proximity is therefore necessary but insufficient for achieving the proximity living ideal; implementation of the 15mC should be behaviourally informed and place-sensitive, coupling abundant local provision of routine needs with access enhancement to specialised amenities to avoid unintended equity trade-offs.

----

## Citation

You can cite this paper at:
```
@online{zhang2025recurrent,
  title = {Recurrent Visitations Expose the Paradox of Human Mobility in the 15-{{Minute City}} Vision},
  author = {Zhang, Xiuning and Poliakov, Alexei and Tenkanen, Henrikki and Arcaute, Elsa},
  date = {2025-08-31},
  eprint = {2509.00919},
  eprinttype = {arXiv},
  eprintclass = {physics},
  doi = {10.48550/arXiv.2509.00919},
  url = {http://arxiv.org/abs/2509.00919},
}
```
