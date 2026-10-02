# idopNetwork proximal causal inference

This repository accompanies **idopNetwork proximal causal inference for molecular exposures and restricted survival**. The method connects molecular network structure to proxy measurements of latent confounding and estimates total protein-exposure effects on linear outcomes, restricted mean survival time (RMST) and survival probability.

Read [the manuscript](manuscript.md) for the model, proofs and results. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) gives study settings, data sources and supporting-study provenance.

## Method

1. Use 75% of patients to learn molecular preprocessing, niche curves and network supports across windows.
2. Construct exposure-specific treatment and outcome proxies under the structural coverage and exposure-boundary conditions.
3. Learn joint readout coordinates and choose a design with complete conditional information.
4. Freeze the molecular design and estimate concentrated bridge moments in the remaining 25% of patients.
5. Use inverse censoring weights for survival targets and intersect Fieller confidence sets for the effect.

The clinical contrast is a one-discovery-standard-deviation change in protein abundance, expressed as months of RMST or percentage points of survival at 36 months. The causal interpretation uses the proxy exclusions, complete bridge representation and censoring conditions stated in the paper.

## Evidence

The component study evaluates ten estimators in seven molecular systems, with 200 datasets per system. In the modular system, linear-effect RMSE is 0.049 for the joint readout, 0.195 for a global-window design and 0.184 for correlation-selected proxies. Reporting counts are 200/200, 191/200 and 200/200. In the weak two-factor system, joint readout reduces linear bias from 0.203 to 0.013 relative to conditional projection of the same design. Complete comparisons include both outcomes and all ten estimators.

A separate fresh-sample study evaluates the complete independent workflow. Coverage is 0.950–0.995 across seven systems and two targets. Systems II, III and V yield 186–190 bounded linear sets per 200 attempts; weak system IV yields 21/200. Point availability, coverage and bounded-set frequency are reported together.

TCGA ovarian cancer and lung adenocarcinoma provide clinical applications. The ovarian PTEN contrast is 1.02 months of 36-month RMST, with a 95% confidence set of [-2.07, 4.24]. The ovarian panel has two bounded RMST sets; the lung panel has none. All 120 protein records and both endpoint sets are supplied.

## Run the current workflow

Use Python 3.10 or later and install `requirements.txt` in an isolated environment. Document building additionally requires `requirements-documents.txt` and the Pandoc executable.

```bash
python -m pip install -r requirements.txt
python workflow.py verify
python workflow.py tables
python workflow.py figures
```

`verify` checks stored simulation records, paired comparisons, bridge algebra and publication consistency without patient data. `tables` refreshes Tables 2–5 from the stored summaries and preserves the manuscript's prose. `figures` regenerates Figures 1, 4, 6 and 7 from current records; the remaining figures are supplied as publication assets.

```bash
# Recompute the final seven-system study (200 datasets per system).
python workflow.py simulate

# Download public input tables, then run both clinical cohorts.
python fetch_tcga.py ov luad
python workflow.py clinical

# Build and check the Word manuscript.
python -m pip install -r requirements-documents.txt
python workflow.py word
```

Simulation and clinical commands write to their declared result directories. Committed records provide the reference run in Git history. Detailed component-study commands and fixed seeds are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Repository layout

| Location | Purpose |
| --- | --- |
| `manuscript.md`, `manuscript_SiM.docx` | Canonical manuscript and generated Word document |
| `workflow.py` | Current analysis, verification and document commands |
| `independent_design_bridge.py` | Discovery-fixed concentrated moment inference |
| `conditional_design_bridge.py`, `joint_readout_bridge.py`, `multiscale_bridge.py` | Proxy design and molecular representation |
| `causal_survival.py` | Censoring weights and survival pseudo-outcomes |
| `results/` | Simulation records, clinical aggregates and source manifests |
| `figures/` | Publication PDF and PNG figures |
| `DISCOVERY_ESTIMATION_PROTOCOL_20261001.md` | Frozen final analysis protocol |
| `release_layout.py` | Explicit publication file inventory and code dependencies |

The paper builder reads canonical Markdown directly. Revision archives, draft manuscripts and incremental manuscript migration scripts are excluded from the publication inventory. Source manifests retain exact code and protocol fingerprints for each study.

## Data and attribution

TCGA RPPA and clinical tables come from cBioPortal's TCGA PanCancer Atlas studies. `fetch_tcga.py` retrieves them into the local `data/` directory; patient input tables are not distributed. See `data/README.md` in the Git repository and the source identifiers in the download script.

The implementation builds on idopNetwork and proximal causal inference. The paper contains methodological and dataset citations. Upstream attribution and licensing are retained in `THIRD_PARTY_NOTICES.md` and `third_party/` in the Git repository.
