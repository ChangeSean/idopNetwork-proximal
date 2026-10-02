# Reproducibility

The published workflow is the 75:25 independent discovery/estimation procedure. `manuscript.md` is the sole prose source; `build_current_manuscript.py` updates its main numerical tables. The Word builder consumes that source, including supplementary methods, tables and proofs.

## Primary studies

| Study | Design | Records under `results/` |
| --- | --- | --- |
| Molecular representation | Seven systems × 200 datasets, ten estimators, two outcomes; seed offset 30000 | `joint_readout_validation_20261001/` |
| Complete independent workflow | Seven systems × 200 fresh datasets, two outcomes; offset 140000; 75:25 split; 100 resamples | `discovery_estimation_validation_20261001/` |
| Primary ovarian application | Split seed 20261201; OV 308/103 discovery/estimation patients; 300 resamples; the same directory retains supplementary LUAD results (264/88) | `discovery_estimation_application_20261001/` |
| Supplementary eight-cohort extension | Same implementation, split seed, 36-month targets and 300 resamples; all eight remaining cohorts | `additional_cohorts_application_20261002/` |

The representation study uses joint-information design selection with molecular roles and dimension fixed during resampling. The independent study uses conditional-information selection and discovery-fixed moment inference. Separate outputs distinguish component performance from complete-workflow performance.

Main-text examples from systems III, IV and VII were selected after component results were available to explain the method. The complete seven-system comparison is retained in `all_summary.csv`; `paired_differences.csv` uses common successful repetitions for each comparison. Bias and RMSE use available point estimates; final confidence-set coverage and bounded-set frequency use every attempted dataset, including unavailable designs returned as whole-line sets.

## Commands

`python workflow.py verify` runs bridge and niche-decomposition algebra checks, final-record/source verification, all component paired comparisons and publication consistency checks. It requires no downloaded patient tables. Source files are stored without Git line-ending conversion so recorded byte-level hashes remain reproducible across platforms.

The representation study can be recomputed with:

```bash
for scenario in 0 1 2 3 4 5 6; do
  python validate_joint_readout_bridge.py --scenario "$scenario" --reps 200 --boot 100 --offset 30000
done
python validate_joint_comparators.py --parts 0 1 2 3 4 5 6
python summarize_joint_readout.py
python select_method_evidence.py
```

For PowerShell, use `0..6 | ForEach-Object { python validate_joint_readout_bridge.py --scenario $_ --reps 200 --boot 100 --offset 30000 }` for the loop. Recompute the independent simulation with `python workflow.py simulate`. For the primary application, run `python fetch_tcga.py ov` followed by `python workflow.py clinical --cohort ov`. Download all ten cohorts with `python fetch_tcga.py blca brca coadread kirc lgg luad ov skcm stad ucec`, then run `python workflow.py clinical --cohort all`; `--cohort remaining` recomputes the eight-cohort extension. Set `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to 1 for consistent resource use. The workflow supplies these defaults to its subprocesses.

`python workflow.py tables` preserves editorial text and regenerates Tables 2–5 and S1. It also exports all completed clinical point estimates with their original confidence sets. `python workflow.py figures` includes Figure 3, which uses downloaded OV inputs to replay the saved discovery split, preprocessing, proxy roles and readout coordinates before drawing the selected networks and ODE curve contributions. Its provenance JSON records the discovery and design fingerprints and confirms that analysis CSVs are unchanged. `python workflow.py word` builds native Word equations and checks all fifteen tables and source equations. Word page rendering is an additional visual check; its latest status is recorded in `results/causal_revision_document_audit.json`. `python workflow.py package` assembles the publication inventory with file hashes.

`niche_ode.py` implements the descriptive molecular curves in Figure 3 and the edge weights in Figures S1–S2. It uses GCV cubic smoothing splines, first-degree shifted Legendre terms integrated along the normalised niche coordinate, and centred, standardised ridge regression with penalty 1 under the sum-of-squares convention. The intrinsic curve combines the intercept, linear niche baseline and own-state group; the supported source contributions sum with it to reconstruct the fitted curve. Network edge signs describe mean cumulative source contributions. Source groups come from the fixed molecular support. The supplied MTODE code guides smoothing, integration and ridge refitting; its separate ADSIHT screening is not used. S1–S2 retain their full-cohort reference supports and proxy roles. The provenance JSON records the ODE settings, reconstruction errors and fingerprints for all three figures. These curve fits do not enter clinical bridge estimation.

## Supporting studies

| Study | Settings | Location under `results/` |
| --- | --- | --- |
| Conditional-information selection | Seven × 200; offset 50000 | `conditional_design_validation_20261001/` |
| Complete construction resampling | Three × 100; offset 70000 | `conditional_design_validation_20261001/` |
| Fixed-design concentrated moments | Seven × 200; offset 100000 | `concentrated_bridge_validation_20261001/` |
| Allocation development | Seven × 200; 50:50 allocation; offset 120000 | `independent_design_validation_20261001/` |
| Full-cohort clinical reference | OV and LUAD full-sample designs and resampling | `conditional_design_application_20261001/` |

The 50:50 study preceded selection of the 75:25 rule. In its weak system, 16/200 fits selected rank one and overall linear coverage was 0.905. The selected allocation was evaluated on the offset-140000 batch. The frozen protocol records this sequence.

Each study directory supplies aggregate results and execution/source manifests. Part files are raw batch outputs; combined files provide the analysis-wide view used by the paper.

## Clinical inputs and interpretation

Inputs are RPPA and clinical survival tables from the ten TCGA PanCancer Atlas study identifiers in `fetch_tcga.py`: BLCA, BRCA, COADREAD, KIRC, LGG, LUAD, OV, SKCM, STAD and UCEC. Discovery determines panel filtering, imputation, protein scales, network roles, rank and readout coordinates. Estimation outcomes enter the survival moments and censoring model. The pathway-boundary JSON specifies GAB2 and KDR analyses. `DISCOVERY_ESTIMATION_PROTOCOL_20261001.md` gives the independent analysis protocol; `additional_cohorts_application_20261002/EXTENSION_PROTOCOL.md` gives the eight-cohort extension protocol, and `extension_plan.json` contains pre-run source and input hashes.

The repository contains cohort-level preprocessing parameters, effect records, role sets and aggregate diagnostics, not patient input rows. All 600 exposure records and 1200 endpoint sets are supplied. `point_estimates.csv` contains all 58 completed coefficients with unchanged confidence sets and flags for coefficients outside the intervention-target ranges. Figure 6 displays all 24 ovarian coefficients; Figure S5 displays all 58 across ten cohorts. Table 5 and Figures 5 and 7 focus on ovarian PTEN, SERPINE1 and CCNE1, chosen after analysis for precision and biological relevance. The other bounded ovarian case, HSPA1A, remains in the complete panel. Point directions are exploratory. Biological exclusions and censoring conditions are given in Sections 4 and 6 of the paper.
