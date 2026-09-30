"""Refresh publication figures without rerunning simulations or bootstrap analyses."""
from pathlib import Path
import hashlib
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import figures as F
from run_application import load_cohort, analyse_cohort, BASIS_ORDER, ODE_RIDGE
from idop_core import get_power_function_samples

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def result_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in RESULTS.glob("*.csv")}


def main():
    before = result_hashes()
    F.save(F.fig_schematic(), "fig1_schematic")
    F.save(F.fig_simulation(
        RESULTS / "sim_coverage_vs_delta.csv",
        RESULTS / "sim_tradeoff.csv",
        RESULTS / "sim_p1_vs_p2.csv"), "fig4_simulation")
    F.save(F.fig_cohorts(RESULTS / "cohort_screen.csv"), "fig5_cohorts")

    # Reconstruct plotting inputs, reusing the previously fitted ODE edge list.
    cohort = load_cohort("ov", "OS", survival=True)
    res = analyse_cohort(cohort, solve_ode=False, survival=True)
    res["qd"] = cohort["qd"]
    res["samples"] = get_power_function_samples(res["params"], cohort["qd"].index, n_samples=100)
    res["edgelist"] = pd.read_csv(RESULTS / "ov_OS_cox_edgelist.csv")
    res["basis_order"] = BASIS_ORDER
    res["ode_ridge"] = ODE_RIDGE
    targets = ["LCK", "CTNNB1", "TP53BP1", "MSH6",
               "CDH2", "CDH1", "STAT5A", "CLDN7"]
    tags = {"LCK": "exposure", "CDH2": "exposure",
            "CTNNB1": "Z for LCK", "TP53BP1": "Z for LCK",
            "MSH6": "child of CTNNB1", "CDH1": "Z for CDH2",
            "STAT5A": "Z for CDH2", "CLDN7": "child of CDH1"}
    F.save(F.fig_causal(res, ["LCK", "CDH2"], targets, tags), "fig3_causal")
    F.save(F.fig_network(res, "LCK"), "figS2_network")
    if result_hashes() != before:
        raise RuntimeError("A numerical result table changed during figure refresh")
    print("Refreshed figures 1, 3, 4, 5 and S2; all result CSV hashes unchanged.")


if __name__ == "__main__":
    main()
