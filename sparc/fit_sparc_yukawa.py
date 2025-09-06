#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_sparc_yukawa.py
-------------------
Batch-fit SPARC "rotmod" rotation curves with a Yukawa/delay gravity model:
    v_model^2(r) = v_baryon^2(r) * [ 1 + alpha_eff * ( 1 + r/lambda ) * exp(-r/lambda) ]

Inputs:
  - Either a ZIP file (e.g., Rotmod_LTG.zip) via --zip, or a directory via --data-dir
  - Files should be SPARC-like 'rotmod' tables (e.g., *_rotmod.dat) with columns for:
      radius (kpc), v_obs (km/s), errV (km/s), Vgas, Vdisk, (optional) Vbul

Outputs (in --out):
  - results.csv : alpha_eff, lambda_kpc, chi2, dof per galaxy
  - per-galaxy plots if --make-plots
  - converted CSVs with standardized columns (radius_kpc, v_obs_kms, v_err_kms, v_disk_kms, v_bulge_kms, v_gas_kms)

Dependencies:
  - numpy, pandas, matplotlib
  - scipy (optional; fitter will fallback to a simple Nelder-Mead-like search if SciPy not installed)

License: MIT
"""
import os, re, io, sys, argparse, zipfile, math, json, tempfile, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Model & fitting
# -------------------------------
def yukawa_boost(r_kpc, alpha_eff, lambda_kpc):
    lam = max(lambda_kpc, 1e-12)
    return 1.0 + alpha_eff * (1.0 + r_kpc/lam) * np.exp(-r_kpc/lam)

def v_model_from_components(r_kpc, v_disk, v_bulge, v_gas, alpha_eff, lambda_kpc):
    v_baryon2 = np.clip(v_disk**2 + v_bulge**2 + v_gas**2, 0, None)
    return np.sqrt(np.clip(v_baryon2 * yukawa_boost(r_kpc, alpha_eff, lambda_kpc), 0, None))

def chisq(params, r, v_obs, v_err, v_disk, v_bulge, v_gas, priors=None, bounds=None):
    alpha_eff, lambda_kpc = float(params[0]), float(params[1])

    penalty = 0.0
    # Bounds: alpha in [0, 2], lambda in [0.1, 100] by default
    a_lo, a_hi = (0.0, 2.0) if bounds is None else bounds.get("alpha", (0.0, 2.0))
    l_lo, l_hi = (0.1, 100.0) if bounds is None else bounds.get("lambda", (0.1, 100.0))
    if not (a_lo <= alpha_eff <= a_hi) or not (l_lo <= lambda_kpc <= l_hi):
        return 1e12

    # Gaussian soft priors if provided
    if priors is not None:
        if "alpha_mu" in priors and "alpha_sig" in priors and priors["alpha_sig"] > 0:
            penalty += ((alpha_eff - priors["alpha_mu"]) / priors["alpha_sig"])**2
        if "lam_mu" in priors and "lam_sig" in priors and priors["lam_sig"] > 0:
            penalty += ((lambda_kpc - priors["lam_mu"]) / priors["lam_sig"])**2

    v_mod = v_model_from_components(r, v_disk, v_bulge, v_gas, alpha_eff, lambda_kpc)
    chi = (v_obs - v_mod) / np.maximum(v_err, 1e-6)
    return float(np.sum(chi**2) + penalty)

def fit_galaxy(r, v_obs, v_err, v_disk, v_bulge, v_gas,
               alpha0=0.4, lam0=8.0, priors=None, bounds=None, maxiter=3000):
    # Try SciPy Nelder-Mead; fallback to a simple random-search + shrinking steps
    try:
        from scipy.optimize import minimize
        res = minimize(lambda p: chisq(p, r, v_obs, v_err, v_disk, v_bulge, v_gas, priors, bounds),
                       x0=np.array([alpha0, lam0], dtype=float),
                       method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-6})
        p_best = res.x
        chi2 = float(res.fun)
        success = bool(res.success)
    except Exception:
        # Fallback: crude direct search
        p = np.array([alpha0, lam0], dtype=float)
        step = np.array([0.2, 2.0], dtype=float)
        best = chisq(p, r, v_obs, v_err, v_disk, v_bulge, v_gas, priors, bounds)
        for _ in range(maxiter):
            cand = p + (np.random.rand(2) - 0.5) * step
            cchi = chisq(cand, r, v_obs, v_err, v_disk, v_bulge, v_gas, priors, bounds)
            if cchi < best:
                p, best = cand, cchi
                step *= 0.98
            else:
                step *= 0.999
        p_best, chi2, success = p, best, True

    # Rough error estimate via local Hessian (finite differences)
    eps = 1e-3
    def second_deriv(i, j):
        dp_i = np.zeros(2); dp_i[i] = eps
        dp_j = np.zeros(2); dp_j[j] = eps
        fpp = chisq(p_best + dp_i + dp_j, r, v_obs, v_err, v_disk, v_bulge, v_gas, priors, bounds)
        fpm = chisq(p_best + dp_i - dp_j, r, v_obs, v_err, v_disk, v_bulge, v_gas, priors, bounds)
        fmp = chisq(p_best - dp_i + dp_j, r, v_obs, v_err, v_disk, v_bulge, v_gas, priors, bounds)
        fmm = chisq(p_best - dp_i - dp_j, r, v_obs, v_err, v_disk, v_bulge, v_gas, priors, bounds)
        return (fpp - fpm - fmp + fmm) / (4*eps*eps)
    try:
        H = np.array([[second_deriv(0,0), second_deriv(0,1)],
                      [second_deriv(1,0), second_deriv(1,1)]])
        cov = np.linalg.inv(H)
        errs = np.sqrt(np.clip(np.diag(cov), 0, None))
    except Exception:
        errs = np.array([np.nan, np.nan])

    return (float(p_best[0]), float(p_best[1])), (float(errs[0]), float(errs[1])), float(chi2), success

# -------------------------------
# Parsing SPARC rotmod-style files
# -------------------------------
def parse_rotmod_to_standard(path):
    """
    Parse SPARC rotmod file into standardized DataFrame with columns:
    radius_kpc, v_obs_kms, v_err_kms, v_disk_kms, v_bulge_kms, v_gas_kms
    """
    # Read raw
    with open(path, "r", errors="ignore") as f:
        lines = f.read().splitlines()

    # Heuristic header detection (look for a '# ...' line with expected tokens)
    header_cols = None
    for i, line in enumerate(lines[:20]):
        if line.strip().startswith("#") and ("Rad" in line and ("Vobs" in line or "Vrot" in line)):
            # remove leading '#', split on whitespace
            header_cols = re.sub(r'^[#\s]+', '', line).strip().split()
            break

    # Read table
    if header_cols is not None:
        df = pd.read_csv(path, delim_whitespace=True, comment="#", header=None, names=header_cols)
    else:
        df = pd.read_csv(path, delim_whitespace=True, comment="#", header=None)
        # Assign typical column names if enough columns exist
        if df.shape[1] >= 6:
            header_cols = ["Rad","Vobs","errV","Vgas","Vdisk","Vbul"]
            header_cols += [f"col{i}" for i in range(6, df.shape[1])]
            df.columns = header_cols[:df.shape[1]]
        else:
            # Give up if too few columns
            raise ValueError("Unrecognized rotmod format (too few columns)")

    # Standardize columns
    cols = [c.lower() for c in df.columns]
    df.columns = cols

    # Required columns
    required = ["rad", "vobs", "errv", "vgas", "vdisk"]
    for rc in required:
        if rc not in df.columns:
            raise ValueError(f"Missing column '{rc}' in {os.path.basename(path)}")
    vbul = df["vbul"] if "vbul" in df.columns else np.zeros(len(df))

    out = pd.DataFrame({
        "radius_kpc": pd.to_numeric(df["rad"], errors="coerce"),
        "v_obs_kms": pd.to_numeric(df["vobs"], errors="coerce"),
        "v_err_kms": pd.to_numeric(df["errv"], errors="coerce"),
        "v_disk_kms": pd.to_numeric(df["vdisk"], errors="coerce"),
        "v_bulge_kms": pd.to_numeric(vbul, errors="coerce") if isinstance(vbul, pd.Series) else vbul,
        "v_gas_kms": pd.to_numeric(df["vgas"], errors="coerce"),
    }).dropna()

    # Basic sanity
    out = out[(out["radius_kpc"] > 0) & (out["v_obs_kms"] > 0) & (out["v_err_kms"] > 0)]
    if len(out) < 6:
        raise ValueError("too few valid rows")

    return out

# -------------------------------
# I/O helpers
# -------------------------------
def ensure_out_dirs(base):
    os.makedirs(base, exist_ok=True)
    os.makedirs(os.path.join(base, "converted"), exist_ok=True)
    os.makedirs(os.path.join(base, "plots"), exist_ok=True)

def save_plot(df, file_tag, out_dir, alpha, lam):
    r = df["radius_kpc"].values
    v_obs = df["v_obs_kms"].values
    v_err = df["v_err_kms"].values
    vd = df["v_disk_kms"].values
    vb = df["v_bulge_kms"].values
    vg = df["v_gas_kms"].values
    v_mod = v_model_from_components(r, vd, vb, vg, alpha, lam)

    plt.figure(figsize=(6,4))
    plt.errorbar(r, v_obs, yerr=v_err, fmt='o', label="Observed")
    plt.plot(r, v_mod, label=f"Model (α={alpha:.2f}, λ={lam:.1f} kpc)")
    plt.plot(r, np.sqrt(vd**2+vb**2+vg**2), linestyle='--', label="Baryonic")
    plt.xlabel("Radius (kpc)"); plt.ylabel("Circular speed (km/s)")
    plt.title(file_tag)
    plt.legend()
    outp = os.path.join(out_dir, "plots", f"{file_tag}.png")
    plt.tight_layout(); plt.savefig(outp); plt.close()
    return outp

# -------------------------------
# Main batch pipeline
# -------------------------------
def batch_fit_from_dir(data_dir, out_dir, priors=None, bounds=None,
                       alpha0=0.4, lam0=8.0, make_plots=False):
    ensure_out_dirs(out_dir)
    conv_dir = os.path.join(out_dir, "converted")
    results = []

    files = sorted([f for f in os.listdir(data_dir) if re.search(r"\.(dat|txt|csv)$", f, re.I)])
    for fn in files:
        path = os.path.join(data_dir, fn)
        file_tag = os.path.splitext(fn)[0]
        try:
            df_std = parse_rotmod_to_standard(path)
            # Save standardized CSV
            csv_out = os.path.join(conv_dir, f"{file_tag}.csv")
            df_std.to_csv(csv_out, index=False)

            r = df_std["radius_kpc"].values
            v_obs = df_std["v_obs_kms"].values
            v_err = df_std["v_err_kms"].values
            vd = df_std["v_disk_kms"].values
            vb = df_std["v_bulge_kms"].values
            vg = df_std["v_gas_kms"].values

            (a_hat, l_hat), (ea, el), chi2, ok = fit_galaxy(
                r, v_obs, v_err, vd, vb, vg,
                alpha0=alpha0, lam0=lam0,
                priors=priors, bounds=bounds
            )
            row = {
                "file": fn,
                "alpha_eff": a_hat, "alpha_eff_err": ea,
                "lambda_kpc": l_hat, "lambda_kpc_err": el,
                "chi2": chi2, "dof": max(len(r)-2, 1),
            }
            results.append(row)

            if make_plots:
                save_plot(df_std, file_tag, out_dir, a_hat, l_hat)

        except Exception as e:
            results.append({"file": fn, "error": str(e)})

    res_df = pd.DataFrame(results)
    res_csv = os.path.join(out_dir, "results.csv")
    res_df.to_csv(res_csv, index=False)
    return res_csv, res_df

def extract_zip_to_temp(zip_path):
    tmpdir = tempfile.mkdtemp(prefix="sparc_rotmod_")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmpdir)
    # If the zip contains a single folder, drill down
    sub = os.listdir(tmpdir)
    if len(sub) == 1 and os.path.isdir(os.path.join(tmpdir, sub[0])):
        return os.path.join(tmpdir, sub[0]), tmpdir
    return tmpdir, tmpdir

# -------------------------------
# CLI
# -------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Batch-fit SPARC rotmod rotation curves with Yukawa/delay model.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", type=str, help="Path to Rotmod_LTG.zip (or similar)")
    src.add_argument("--data-dir", type=str, help="Directory with rotmod files (.dat/.txt/.csv)")
    ap.add_argument("--out", type=str, required=True, help="Output directory")
    ap.add_argument("--alpha-mu", type=float, default=0.5, help="Gaussian prior mean for alpha_eff")
    ap.add_argument("--alpha-sig", type=float, default=0.5, help="Gaussian prior sigma for alpha_eff")
    ap.add_argument("--lam-mu", type=float, default=10.0, help="Gaussian prior mean for lambda (kpc)")
    ap.add_argument("--lam-sig", type=float, default=10.0, help="Gaussian prior sigma for lambda (kpc)")
    ap.add_argument("--alpha0", type=float, default=0.4, help="Initial guess alpha_eff")
    ap.add_argument("--lam0", type=float, default=8.0, help="Initial guess lambda (kpc)")
    ap.add_argument("--alpha-bounds", type=str, default="0,2", help="Bounds for alpha_eff, e.g. '0,2'")
    ap.add_argument("--lambda-bounds", type=str, default="0.1,100", help="Bounds for lambda_kpc, e.g. '1,50'")
    ap.add_argument("--make-plots", action="store_true", help="Save per-galaxy fit plots")
    ap.add_argument("--keep-temp", action="store_true", help="Keep temporary extracted folder (when using --zip)")
    return ap.parse_args()

def main():
    args = parse_args()
    priors = {"alpha_mu": args.alpha_mu, "alpha_sig": args.alpha_sig,
              "lam_mu": args.lam_mu, "lam_sig": args.lam_sig}

    # Bounds parsing
    try:
        a_lo, a_hi = [float(x) for x in args.alpha_bounds.split(",")]
        l_lo, l_hi = [float(x) for x in args.lambda_bounds.split(",")]
        bounds = {"alpha": (a_lo, a_hi), "lambda": (l_lo, l_hi)}
    except Exception:
        bounds = {"alpha": (0.0, 2.0), "lambda": (0.1, 100.0)}

    os.makedirs(args.out, exist_ok=True)

    data_dir = None
    tmp_root = None
    try:
        if args.zip:
            data_dir, tmp_root = extract_zip_to_temp(args.zip)
        else:
            data_dir = args.data_dir

        res_csv, res_df = batch_fit_from_dir(
            data_dir, args.out, priors=priors, bounds=bounds,
            alpha0=args.alpha0, lam0=args.lam0, make_plots=args.make_plots
        )

        # Quick summary to stdout & JSON
        ok = res_df.dropna(subset=["alpha_eff","lambda_kpc"]) if "alpha_eff" in res_df.columns else pd.DataFrame()
        summary = {
            "num_total": int(res_df.shape[0]),
            "num_success": int(ok.shape[0]),
            "alpha_eff_median": float(np.nanmedian(ok["alpha_eff"])) if ok.shape[0] else None,
            "lambda_kpc_median": float(np.nanmedian(ok["lambda_kpc"])) if ok.shape[0] else None,
            "results_csv": res_csv
        }
        print(json.dumps(summary, indent=2))

    finally:
        if args.zip and tmp_root and (not args.keep_temp):
            shutil.rmtree(tmp_root, ignore_errors=True)

if __name__ == "__main__":
    main()
