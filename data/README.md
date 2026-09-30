# TCGA PanCancer Atlas data sources

The analysis uses tables obtained from the public cBioPortal API at https://www.cbioportal.org/api. Patient-level tables are downloaded locally using `fetch_tcga.py` and are excluded from the repository.

Each `tcga_<cohort>_rppa_long.csv` corresponds to the molecular profile `<cohort>_tcga_pan_can_atlas_2018_rppa`. Each `tcga_<cohort>_clinical.csv` contains patient-level clinical attributes from that study. The RPPA long tables have `patientId`, `sampleId`, `entrezGeneId`, `value` and `gene` columns. `tcga_pancan_rppa_survey.csv` records the RPPA availability survey.

The cohort collection comprises ovarian serous carcinoma (`ov`), lung adenocarcinoma (`luad`), bladder urothelial carcinoma (`blca`), stomach adenocarcinoma (`stad`), colorectal adenocarcinoma (`coadread`), breast invasive carcinoma (`brca`), kidney renal clear cell carcinoma (`kirc`), lower-grade glioma (`lgg`), skin cutaneous melanoma (`skcm`) and uterine corpus endometrial carcinoma (`ucec`).

The local OV source files were created on 11 September 2026; the LUAD source files were created on 12 September 2026. A cBioPortal revision/commit identifier was not captured during the API download. Checksums of the original analysis inputs are recorded in `SOURCE_DATA.sha256`, independently of later changes at cBioPortal. These checksums do not include patient-level records.

To obtain fresh copies of all ten cohorts, run from the repository root:

```bash
python fetch_tcga.py ov luad blca stad coadread brca kirc lgg skcm ucec
```

Downloading updates replaces the matching input CSV files and can change results if the source data have been revised. The download script also generates pooled RPPA/clinical tables, which are not required for the ten-cohort analysis.

**Dataset citation:** cBioPortal for Cancer Genomics. *TCGA PanCancer Atlas Studies* [dataset]. Memorial Sloan Kettering Cancer Center; 2018. https://www.cbioportal.org/datasets.

Source data remain subject to the source providers' terms; the upstream idop software license does not apply to these datasets.
