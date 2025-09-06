#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
postprocess_results.py
----------------------
Read a results.csv from the fitter and produce summary stats + quick plots + LaTeX tables.

Usage:
  python postprocess_results.py --results ./out/results.csv --out ./out
"""
import os, argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Path to results.csv")
    ap.add_argument("--out", required=True, help="Output directory (plots/TEX here)")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.results)
    ok = df.dropna(subset=["alpha_eff","lambda_kpc"]) if "alpha_eff" in df.columns else pd.DataFrame()

    # Reduced chi^2 if available
    if "chi2_red" not in ok.columns and "chi2" in ok.columns and "dof" in ok.columns:
        ok["chi2_red"] = ok["chi2"] / ok["dof"].replace(0, np.nan)

    summary = {
        "num_total": int(df.shape[0]),
        "num_success": int(ok.shape[0]),
        "alpha_mean": float(np.nanmean(ok["alpha_eff"])) if ok.shape[0] else None,
        "alpha_median": float(np.nanmedian(ok["alpha_eff"])) if ok.shape[0] else None,
        "alpha_min": float(np.nanmin(ok["alpha_eff"])) if ok.shape[0] else None,
        "alpha_max": float(np.nanmax(ok["alpha_eff"])) if ok.shape[0] else None,
        "lambda_mean": float(np.nanmean(ok["lambda_kpc"])) if ok.shape[0] else None,
        "lambda_median": float(np.nanmedian(ok["lambda_kpc"])) if ok.shape[0] else None,
        "lambda_min": float(np.nanmin(ok["lambda_kpc"])) if ok.shape[0] else None,
        "lambda_max": float(np.nanmax(ok["lambda_kpc"])) if ok.shape[0] else None,
        "chi2_median": float(np.nanmedian(ok["chi2"])) if "chi2" in ok.columns else None,
        "chi2red_median": float(np.nanmedian(ok["chi2_red"])) if "chi2_red" in ok.columns else None,
        "frac_alpha_at_upper": float(np.mean(ok["alpha_eff"] >= ok["alpha_eff"].max())) if ok.shape[0] else None
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Plots
    plt.figure(figsize=(5,3.5)); plt.hist(ok["alpha_eff"].values, bins=25); plt.xlabel("alpha_eff"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "alpha_hist.png")); plt.close()
    plt.figure(figsize=(5,3.5)); plt.hist(ok["lambda_kpc"].values, bins=25); plt.xlabel("lambda (kpc)"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "lambda_hist.png")); plt.close()
    plt.figure(figsize=(4.5,4.5)); plt.scatter(ok["lambda_kpc"].values, ok["alpha_eff"].values, s=10)
    plt.xlabel("lambda (kpc)"); plt.ylabel("alpha_eff")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "alpha_vs_lambda.png")); plt.close()

    # LaTeX table (compact)
    tex_path = os.path.join(args.out, "results_table.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[t]\\centering\\small\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Statistic & $\\alpha_{\\rm eff}$ mean & median & $\\lambda$ mean [kpc] & median \\\\\n\\midrule\n")
        f.write(f"All galaxies & {summary['alpha_mean']:.3f} & {summary['alpha_median']:.3f} & {summary['lambda_mean']:.1f} & {summary['lambda_median']:.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\caption{Summary of Yukawa/delay fits.}\n\\end{table}\n")
    print("Wrote:", tex_path)

if __name__ == "__main__":
    main()
