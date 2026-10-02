# Eight-cohort extension of the fixed clinical analysis

Analysis initiated on 2026-10-02 after the user requested all remaining cohorts.
The included cohorts are BLCA, BRCA, COADREAD, KIRC, LGG, SKCM, STAD and UCEC.
Each cohort is analysed once with the existing final clinical implementation.

The split is 75:25 with seed 20261201. Discovery-only preprocessing selects
60 proteins, network candidates, proxy roles, bridge dimension and readout
coordinates. Estimation uses 300 patient resamples with seed 20261301,
censoring refits and the same 36-month RMST and survival targets. The exposure
unit is one discovery-cohort standard deviation. All 60 attempted proteins
and all confidence-set shapes are retained for each cohort.

The implementation and settings are unchanged from the final OV/LUAD analysis.
The batch runner changes only the output directory and distributes independent
cohorts across processes. Source and input file hashes are recorded before
analysis in extension_plan.json. Original OV/LUAD outputs are preserved.

The confidence sets use the existing per-exposure inference. Exclusion of zero
is reported separately from a directional finite interval; scanning proteins,
endpoints and cohorts does not create multiplicity-adjusted discoveries.
