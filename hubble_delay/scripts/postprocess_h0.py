#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
postprocess_h0.py
-----------------
Read results.json and produce a compact LaTeX table + quick plots if present.
"""
import os, json, argparse
import matplotlib.pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Path to results.json")
    ap.add_argument("--out", required=True, help="Output directory for table/plots")
    return ap.parse_args()

def main():
    args = parse_args()
    with open(args.results, "r") as f:
        res = json.load(f)
    # Write a LaTeX table
    tex = os.path.join(args.out, "h0_table.tex")
    with open(tex, "w") as f:
        f.write("\\begin{table}[t]\\centering\\small\n")
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("$H_0$ & $\\Omega_m$ & $\\Omega_k$ & $w_0$ & $w_a$ & $r_d$~[Mpc] & $\\chi^2_\\nu$\\\\\n\\midrule\n")
        f.write(f"{res['H0']:.2f} & {res['Om']:.3f} & {res['Ok']:.3f} & {res['w0']:.2f} & {res['wa']:.2f} & {res['rd']:.1f} & {res.get('chi2_red', float('nan')):.2f} \\\\\n")
        f.write("\\bottomrule\n\\caption{Best-fit parameters for the delay/CPL background.}\n\\end{tabular}\n\\end{table}\n")
    print("Wrote:", tex)

if __name__ == "__main__":
    main()
