# idopNetwork–proximal

Code and reproducibility materials for **Network-guided proximal causal inference with idopNetwork**.

The method connects idop population curves and patient deviations, stability-selected molecular networks, exposure-specific treatment and outcome proxies, and a reduced-rank proximal outcome bridge. The linear bridge estimates total intervention contrasts under the manuscript's latent-factor and structural proxy conditions. A Cox second stage supplies a survival regression extension.

## Installation

Python 3.10 or newer is recommended. Clone the repository and create an isolated environment:

```bash
git clone https://github.com/ChangeSean/idopNetwork-proximal.git
cd idopNetwork-proximal
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on macOS/Linux, then install the analysis dependencies:

```bash
python -m pip install -r requirements.txt
```

`requirements-verified.txt` records the installed analysis dependency versions used to verify this upload. To use those versions instead, run `python -m pip install -r requirements-verified.txt`.

## Analysis workflow

```text
protein levels X, ordered by the niche index
  -> protein-specific power curves C(s)
  -> patient deviations U = X - C(s)
  -> SVD latent patient space, loadings and rank
  -> nodewise LASSO support graph, selected across niche windows
  -> exposure-specific Z (neighbours) and W (separate components)
  -> reduced-rank proximal bridge or Cox second stage
  -> estimates, standard errors, proxy strength and reporting diagnostics
```

The support graph and latent representation are fitted once per cohort; proxy sets and bridge fits change with the exposure. The weak-form ODE gives descriptive signed edge weights and curve decompositions.

## Reproduce the TCGA analyses

Download the public cBioPortal RPPA and clinical data before running the analyses:

```bash
python fetch_tcga.py ov luad blca stad coadread brca kirc lgg skcm ucec
```

The tables are written to `data/` locally. Patient-level tables are not included in this repository. See `data/README.md` for the exact study/profile identifiers. Then run:

```bash
python run_application.py ov OS --survival --boot 300 --ode
python run_application.py luad OS --survival --boot 300
python run_application.py ov PFS --survival --boot 300
python run_application.py ov DFS --survival --boot 300
python run_application.py luad PFS --survival --boot 300
python screen_cohorts.py
```

The main settings are 60 most variable proteins, `alpha=0.15`, five windows, edge-selection frequency 0.6, proxy-strength floor 0.05, and a truncation cap of estimated latent rank plus one. Use `--p`, `--alpha`, `--k` and `--thr` to specify other settings. Patient bootstrap refits the first stage, projection and Cox stage with proxy roles fixed.

For the binary linear-outcome analysis, use `python run_application.py ov PFS` without `--survival`.

The stored `results/` files retain the original estimates and bootstrap summaries. Rerunning a command writes new output to the corresponding result files; preserve the supplied results if comparing a rerun with the original analysis.

## Reproduce simulations and figures

```bash
python simulate.py all
python simulate.py violations 1000 100
python simulate.py safeguards 1000 100
python simulate.py ablation 1000 60
python simulate.py breadth
python simulate.py survival
python make_figures.py ov OS
```

| Command | Output and purpose |
|---|---|
| `simulate.py all` | Total-effect coverage versus structural perturbation (Table 2), latent-signal/network/proxy-strength trade-off (Table 3) |
| `simulate.py violations` | Separate P1 and P2 perturbations in Figure 4 |
| `simulate.py safeguards` | Truncation rank and proxy-strength threshold settings (Table 4) |
| `simulate.py ablation` | Curve deviations, niche ordering and window comparison |
| `simulate.py breadth` | Sample size and latent rank grid (Table S2) |
| `simulate.py survival` | Survival extension, auxiliary Cox reference and bootstrap comparison (Table S3) |
| `make_figures.py ov OS` | Figures 1–7, S1 and S2; recomputes cohort fits, bootstrap estimates and stability scans |
| `refresh_figures_from_results.py` | Refreshes figures from saved result tables; requires the supplied cohort data |

Simulation modes `all`, `coverage`, `tradeoff`, `violations`, `safeguards`, `ablation` and `survival` accept sample size and replicate count as the second and third arguments. For example, `python simulate.py coverage 1000 100`. The `breadth` mode uses its built-in sample-size grid; `python simulate.py breadth 1000 60` specifies 60 repeats per setting.

Simulation random seeds are set in `simulate.py`. Coverage summaries are conditional on reported estimates, and reporting/abstention rates are stored alongside them. Full simulations and bootstrap analyses may take substantial time depending on hardware. The survival simulation compares intervals with an auxiliary reference fitted separately within each replicate; its inclusion rate is distinct from coverage of a fixed population parameter.

## Files

| File/directory | Contents |
|---|---|
| `idop_core.py` | Transformations, niche ordering, power curves, stable support selection and weak-form ODE |
| `proximal.py` | Latent representation, proxy construction, linear bridge, proxy strength and BH adjustment |
| `survival.py` | Breslow Cox model and proximal Cox second stage |
| `run_application.py` | Shared cohort analysis implementation and application CLI |
| `dgp.py`, `simulate.py` | Data-generating system and simulation studies |
| `fetch_tcga.py` | Public cBioPortal API download |
| `screen_cohorts.py` | Cohort-level network and proxy diagnostics |
| `figures.py`, `make_figures.py` | Publication figure generation |
| `refresh_figures_from_results.py` | Figure refresh using saved results |
| `data/README.md` | Data source, study/profile identifiers and download instructions |
| `results/` | Original application, network, stability and simulation CSV results |
| `figures/` | Seven main and two supplementary figures, as PDF and PNG |
| `third_party/` | Retained upstream idopNetwork license |

## Data source and citation

The data are from cBioPortal's **TCGA PanCancer Atlas** collection, with study IDs `<cohort>_tcga_pan_can_atlas_2018`. The ten analysed cohort prefixes are `ov`, `luad`, `blca`, `stad`, `coadread`, `brca`, `kirc`, `lgg`, `skcm` and `ucec`. See [data/README.md](data/README.md) for provenance and download instructions. Input-data checksums are included there as `SOURCE_DATA.sha256`; downloaded copies may differ if cBioPortal has revised its data since the original analysis.

**Dataset citation:** cBioPortal for Cancer Genomics. *TCGA PanCancer Atlas Studies* [dataset]. Memorial Sloan Kettering Cancer Center; 2018. https://www.cbioportal.org/datasets.

Please also cite the source methodology and data resources:

- Miao W, Geng Z, Tchetgen Tchetgen EJ. Identifying causal effects with proxy variables of an unmeasured confounder. *Biometrika*. 2018;105:987–993.
- Chen C, et al. An omnidirectional visualization model of personalized gene regulatory networks. *npj Systems Biology and Applications*. 2019;5:38.
- Dong A, et al. idopNetwork: a network tool to dissect spatial community ecology. *Methods in Ecology and Evolution*. 2023;14:2272–2283.
- Cerami E, et al. The cBio Cancer Genomics Portal: an open platform for exploring multidimensional cancer genomics data. *Cancer Discovery*. 2012;2:401–404.
- Li J, et al. TCPA: a resource for cancer functional proteomics data. *Nature Methods*. 2013;10:1046–1047.

## Attribution and licensing

The idop core retains the attribution to Yu Wang and the upstream MIT notice in [third_party/LICENSE.idopnetwork](third_party/LICENSE.idopnetwork). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). That notice applies to upstream-derived code. The repository does not assign a new blanket license to the manuscript's original proximal extensions, figures or source TCGA datasets.
