# Reproducibility

The published workflow is the 75:25 independent discovery/estimation procedure. `manuscript.md` is the sole prose source; `build_current_manuscript.py` updates its main numerical tables. The Word builder consumes that source, including supplementary methods, tables and proofs.

## Primary studies

| Study | Design | Records under `results/` |
| --- | --- | --- |
| Molecular representation | Seven systems × 200 datasets, ten estimators, two outcomes; seed offset 30000 | `joint_readout_validation_20261001/` |
| Complete independent workflow | Seven systems × 200 fresh datasets, two outcomes; offset 140000; 75:25 split; 100 resamples | `discovery_estimation_validation_20261001/` |
| Clinical application | One split per cohort, seed 20261201; OV 308/103 and LUAD 264/88 discovery/estimation patients; 300 resamples | `discovery_estimation_application_20261001/` |
| Eight-cohort extension | Same implementation, split seed, 36-month targets and 300 resamples; all eight remaining cohorts | `additional_cohorts_application_20261002/` |

The representation study uses joint-information design selection and evaluates components with molecular roles and dimension fixed during resampling. The independent study evaluates conditional-information selection and discovery-fixed moment inference. These are distinct experiments. Their results are not pooled.

Main-text examples from systems III, IV and VII were selected after component results were available to explain the method. The complete seven-system comparison is retained in `all_summary.csv`; `paired_differences.csv` uses common successful repetitions for each comparison. Bias and RMSE use available point estimates; final confidence-set coverage and bounded-set frequency use every attempted dataset, including unavailable designs returned as whole-line sets.

## Commands

`python workflow.py verify` runs bridge algebra checks, final-record/source verification, all component paired comparisons and publication consistency checks. It requires no downloaded patient tables. Source files are stored without Git line-ending conversion so recorded byte-level hashes remain reproducible across platforms.

The representation study can be recomputed with:

```bash
for scenario in 0 1 2 3 4 5 6; do
  python validate_joint_readout_bridge.py --scenario "$scenario" --reps 200 --boot 100 --offset 30000
done
python validate_joint_comparators.py --parts 0 1 2 3 4 5 6
python summarize_joint_readout.py
python select_method_evidence.py
```

For PowerShell, use `0..6 | ForEach-Object { python validate_joint_readout_bridge.py --scenario $_ --reps 200 --boot 100 --offset 30000 }` for the loop. The final study is `python workflow.py simulate`. Download all ten cohorts with `python fetch_tcga.py blca brca coadread kirc lgg luad ov skcm stad ucec`, then run `python workflow.py clinical --cohort all`. Use `--cohort remaining` to recompute only the eight-cohort extension. Set `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to 1 for consistent resource use. The workflow supplies these defaults to its subprocesses.

`python workflow.py tables` preserves editorial text and regenerates Tables 2–5 and S1. It also exports all completed clinical point estimates with their original confidence sets. `python workflow.py word` builds native Word equations and checks all fifteen tables and source equations. Word page rendering is an additional visual check; its latest status is recorded in `results/causal_revision_document_audit.json`. `python workflow.py package` assembles the publication inventory with file hashes.

## Supporting studies

| Study | Settings | Location under `results/` |
| --- | --- | --- |
| Conditional-information selection | Seven × 200; offset 50000 | `conditional_design_validation_20261001/` |
| Complete construction resampling | Three × 100; offset 70000 | `conditional_design_validation_20261001/` |
| Fixed-design concentrated moments | Seven × 200; offset 100000 | `concentrated_bridge_validation_20261001/` |
| Allocation development | Seven × 200; 50:50 allocation; offset 120000 | `independent_design_validation_20261001/` |
| Full-cohort clinical reference | OV and LUAD full-sample designs and resampling | `conditional_design_application_20261001/` |

The 50:50 study preceded selection of the fixed 75:25 rule. In its weak system, 16/200 fits selected rank one and overall linear coverage was 0.905. The final rule was evaluated on the separate offset-140000 batch. The frozen protocol records this sequence. Supporting studies occupy one supplementary-methods section and are not presented as repeated confirmations of the final estimator.

Each study directory supplies aggregate results and execution/source manifests. Part files are raw batch outputs; combined files provide the analysis-wide view used by the paper. Earlier Git commits preserve the original Cox-regression release, superseded as the primary workflow by the intervention-mean survival bridge.

## Clinical inputs and interpretation

Inputs are RPPA and clinical survival tables from the ten TCGA PanCancer Atlas study identifiers in `fetch_tcga.py`: BLCA, BRCA, COADREAD, KIRC, LGG, LUAD, OV, SKCM, STAD and UCEC. Discovery alone determines panel filtering, imputation, protein scales, network roles, rank and readout coordinates. Estimation outcomes enter the survival moments and censoring model. The pathway-boundary JSON specifies separate GAB2 and KDR analyses. The original frozen protocol remains unchanged; `additional_cohorts_application_20261002/EXTENSION_PROTOCOL.md` records the extension requested on 2026-10-02, and `extension_plan.json` contains pre-run source and input hashes.

The repository contains cohort-level preprocessing parameters, effect records, role sets and aggregate diagnostics, not patient input rows. All 600 exposure records and 1200 endpoint sets are supplied. `point_estimates.csv` contains all 58 completed coefficients with unchanged confidence sets and flags for coefficients outside the intervention-target ranges; Figure 6 displays all 58. Worked examples in Table 5 and Figure 7 were chosen after analysis to show both bounded OV cases, PTEN in two cohorts and a completed LUAD contrast within the target ranges. Point directions are exploratory. Biological exclusions and censoring conditions are given in Sections 4 and 6 of the paper.
