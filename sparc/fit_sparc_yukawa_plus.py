#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_sparc_yukawa_plus.py
------------------------
Enhanced batch-fitting for SPARC rotmod rotation curves with a Yukawa/delay model.
Adds optional:
  - per-galaxy M/L scaling (disk, bulge) as nuisance params with Gaussian priors,
  - intrinsic error floor s_int (km/s) added in quadrature,
  - robust loss (Huber) to downweight outliers,
  - bounds control, priors control via CLI.

Model:
  v_model^2 = (Υ_d * v_disk^2 + Υ_b * v_bulge^2 + v_gas^2) * [ 1 + α * (1 + r/λ) * exp(-r/λ) ]

Usage:
  python fit_sparc_yukawa_plus.py --zip Rotmod_LTG.zip --out ./out_plus --make-plots \
      --fit-ml --ml-prior 1.0,0.3 --fit-errfloor --robust --lambda-bounds 1,50 --alpha-bounds 0,5
"""
import os, re, io, sys, argparse, zipfile, tempfile, shutil, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- Model pieces --------------------
def yukawa_boost(r_kpc, alpha_eff, lambda_kpc):
    lam = max(lambda_kpc, 1e-12)
    return 1.0 + alpha_eff * (1.0 + r_kpc/lam) * np.exp(-r_kpc/lam)

def v_model(r, v_disk, v_bulge, v_gas, alpha_eff, lambda_kpc, y_disk=1.0, y_bul=1.0):
    v2_bar = np.clip(y_disk*(v_disk**2) + y_bul*(v_bulge**2) + v_gas**2, 0, None)
    return np.sqrt(np.clip(v2_bar * yukawa_boost(r, alpha_eff, lambda_kpc), 0, None))

def huber_residuals(res, delta):
    # Huber loss: quadratic for |res|<=delta, linear beyond.
    a = np.abs(res)
    w = np.ones_like(a)
    mask = a > delta
    w[mask] = delta / a[mask]
    return res * w

# -------------------- Parsing --------------------
def parse_rotmod_to_standard(path):
    with open(path, "r", errors="ignore") as f:
        lines = f.read().splitlines()
    header_cols = None
    for i, line in enumerate(lines[:20]):
        if line.strip().startswith("#") and ("Rad" in line and ("Vobs" in line or "Vrot" in line)):
            header_cols = re.sub(r'^[#\s]+', '', line).strip().split()
            break
    if header_cols is not None:
        df = pd.read_csv(path, delim_whitespace=True, comment="#", header=None, names=header_cols)
    else:
        df = pd.read_csv(path, delim_whitespace=True, comment="#", header=None)
        if df.shape[1] >= 6:
            header_cols = ["Rad","Vobs","errV","Vgas","Vdisk","Vbul"]
            header_cols += [f"col{i}" for i in range(6, df.shape[1])]
            df.columns = header_cols[:df.shape[1]]
        else:
            raise ValueError("Unrecognized rotmod format (too few columns)")
    cols = [c.lower() for c in df.columns]
    df.columns = cols
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
    out = out[(out["radius_kpc"]>0) & (out["v_obs_kms"]>0) & (out["v_err_kms"]>0)]
    if len(out) < 6:
        raise ValueError("too few valid rows")
    return out

# -------------------- Fitting --------------------
def chisq(params, r, v_obs, v_err, v_disk, v_bulge, v_gas, cfg):
    # params layout:
    #   [alpha, lambda, (optional) y_disk, (optional) y_bulge, (optional) s_int]
    i = 0
    alpha = params[i]; i+=1
    lam = params[i]; i+=1
    y_d = cfg["y_disk_init"]; y_b = cfg["y_bul_init"]; s_int = 0.0
    if cfg["fit_ml"]:
        y_d = params[i]; i+=1
        y_b = params[i]; i+=1
    if cfg["fit_errfloor"]:
        s_int = params[i]; i+=1

    # Bound checks
    a_lo, a_hi = cfg["alpha_bounds"]; l_lo, l_hi = cfg["lambda_bounds"]
    if not (a_lo <= alpha <= a_hi and l_lo <= lam <= l_hi):
        return 1e12
    if cfg["fit_ml"]:
        yd_lo, yd_hi = cfg["yd_bounds"]; yb_lo, yb_hi = cfg["yb_bounds"]
        if not (yd_lo <= y_d <= yd_hi and yb_lo <= y_b <= yb_hi):
            return 1e12
    if cfg["fit_errfloor"]:
        if not (0.0 <= s_int <= cfg["sint_max"]):
            return 1e12

    v_mod = v_model(r, v_disk, v_bulge, v_gas, alpha, lam, y_d, y_b)
    sig = np.sqrt(v_err**2 + s_int**2)

    res = (v_obs - v_mod) / np.maximum(sig, 1e-6)
    if cfg["robust"]:
        res = huber_residuals(res, cfg["huber_delta"])

    chi2 = float(np.sum(res**2))

    # Gaussian priors
    if cfg["priors"]:
        pr = cfg["priors"]
        if pr.get("alpha_sig", 0)>0:
            chi2 += ((alpha - pr["alpha_mu"]) / pr["alpha_sig"])**2
        if pr.get("lam_sig", 0)>0:
            chi2 += ((lam - pr["lam_mu"]) / pr["lam_sig"])**2
        if cfg["fit_ml"] and pr.get("ml_sig", 0)>0:
            chi2 += ((y_d - pr["ml_mu"]) / pr["ml_sig"])**2
            chi2 += ((y_b - pr["ml_mu"]) / pr["ml_sig"])**2
        if cfg["fit_errfloor"] and pr.get("sint_sig", 0)>0:
            chi2 += ((s_int - pr["sint_mu"]) / pr["sint_sig"])**2

    return chi2

def fit_one(df, cfg):
    r = df["radius_kpc"].values
    v_obs = df["v_obs_kms"].values
    v_err = df["v_err_kms"].values
    vd = df["v_disk_kms"].values
    vb = df["v_bulge_kms"].values
    vg = df["v_gas_kms"].values

    p0 = [cfg["alpha0"], cfg["lam0"]]
    if cfg["fit_ml"]: p0 += [cfg["y_disk_init"], cfg["y_bul_init"]]
    if cfg["fit_errfloor"]: p0 += [cfg["sint_init"]]

    try:
        from scipy.optimize import minimize
        res = minimize(lambda p: chisq(p, r, v_obs, v_err, vd, vb, vg, cfg),
                       x0=np.array(p0, dtype=float), method="Nelder-Mead",
                       options={"maxiter": cfg["maxiter"], "xatol": 1e-6, "fatol": 1e-6})
        p_best = res.x; chi2 = float(res.fun); ok = bool(res.success)
    except Exception:
        # Fallback random search
        p = np.array(p0, dtype=float); step = np.array([0.2, 2.0] + ([0.2,0.2] if cfg["fit_ml"] else []) + ([1.0] if cfg["fit_errfloor"] else []))
        best = chisq(p, r, v_obs, v_err, vd, vb, vg, cfg); ok=True
        for _ in range(cfg["maxiter"]):
            cand = p + (np.random.rand(len(p))-0.5)*step
            cchi = chisq(cand, r, v_obs, v_err, vd, vb, vg, cfg)
            if cchi < best: p, best = cand, cchi; step *= 0.98
            else: step *= 0.999
        p_best, chi2 = p, best

    # Unpack
    i=0
    alpha = float(p_best[i]); i+=1
    lam = float(p_best[i]); i+=1
    y_d = cfg["y_disk_init"]; y_b = cfg["y_bul_init"]; s_int = 0.0
    if cfg["fit_ml"]:
        y_d = float(p_best[i]); i+=1
        y_b = float(p_best[i]); i+=1
    if cfg["fit_errfloor"]:
        s_int = float(p_best[i]); i+=1

    dof = max(len(r) - (2 + (2 if cfg["fit_ml"] else 0) + (1 if cfg["fit_errfloor"] else 0)), 1)
    red = chi2 / dof if dof>0 else np.nan
    return {"alpha_eff": alpha, "lambda_kpc": lam, "y_disk": y_d, "y_bulge": y_b, "s_int": s_int,
            "chi2": chi2, "chi2_red": red, "dof": dof}

# -------------------- I/O & Pipeline --------------------
def ensure_out_dirs(base):
    os.makedirs(base, exist_ok=True)
    os.makedirs(os.path.join(base, "converted"), exist_ok=True)
    os.makedirs(os.path.join(base, "plots"), exist_ok=True)

def save_plot(df, tag, out_dir, fit):
    r = df["radius_kpc"].values
    v_obs = df["v_obs_kms"].values
    v_err = df["v_err_kms"].values
    vd = df["v_disk_kms"].values
    vb = df["v_bulge_kms"].values
    vg = df["v_gas_kms"].values
    v_mod = v_model(r, vd, vb, vg, fit["alpha_eff"], fit["lambda_kpc"], fit["y_disk"], fit["y_bulge"])
    plt.figure(figsize=(6,4))
    plt.errorbar(r, v_obs, yerr=v_err, fmt='o', label="Observed")
    plt.plot(r, v_mod, label=f"Model (α={fit['alpha_eff']:.2f}, λ={fit['lambda_kpc']:.1f} kpc)")
    plt.plot(r, np.sqrt(fit["y_disk"]*vd**2 + fit["y_bulge"]*vb**2 + vg**2), '--', label="Baryonic (scaled)")
    plt.xlabel("Radius (kpc)"); plt.ylabel("Circular speed (km/s)")
    plt.title(tag); plt.legend()
    outp = os.path.join(out_dir, "plots", f"{tag}.png")
    plt.tight_layout(); plt.savefig(outp); plt.close()
    return outp

def extract_zip_to_temp(zip_path):
    tmpdir = tempfile.mkdtemp(prefix="sparc_rotmod_")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmpdir)
    sub = os.listdir(tmpdir)
    if len(sub)==1 and os.path.isdir(os.path.join(tmpdir, sub[0])):
        return os.path.join(tmpdir, sub[0]), tmpdir
    return tmpdir, tmpdir

# -------------------- CLI --------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Enhanced SPARC Yukawa/delay fits with M/L, error floor, robust loss.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", type=str, help="Path to Rotmod_LTG.zip")
    src.add_argument("--data-dir", type=str, help="Directory with rotmod files (.dat/.txt/.csv)")
    ap.add_argument("--out", type=str, required=True, help="Output directory")
    # Base priors
    ap.add_argument("--alpha-mu", type=float, default=0.5)
    ap.add_argument("--alpha-sig", type=float, default=0.5)
    ap.add_argument("--lam-mu", type=float, default=10.0)
    ap.add_argument("--lam-sig", type=float, default=10.0)
    # Bounds
    ap.add_argument("--alpha-bounds", type=str, default="0,2")
    ap.add_argument("--lambda-bounds", type=str, default="0.1,100")
    # Initial guesses
    ap.add_argument("--alpha0", type=float, default=0.4)
    ap.add_argument("--lam0", type=float, default=8.0)
    # M/L options
    ap.add_argument("--fit-ml", action="store_true", help="Fit disk/bulge M/L scalings Υ_d, Υ_b")
    ap.add_argument("--ml-prior", type=str, default="1.0,0.3", help="mean,sigma for Υ prior")
    ap.add_argument("--ml-bounds", type=str, default="0.3,2.0", help="bounds for Υ: min,max")
    # Error floor
    ap.add_argument("--fit-errfloor", action="store_true", help="Fit intrinsic error floor s_int [km/s]")
    ap.add_argument("--sint-init", type=float, default=5.0)
    ap.add_argument("--sint-max", type=float, default=20.0)
    # Robust loss
    ap.add_argument("--robust", action="store_true", help="Use Huber loss")
    ap.add_argument("--huber-delta", type=float, default=10.0, help="Huber delta in (v/sigma) units")
    ap.add_argument("--make-plots", action="store_true")
    ap.add_argument("--maxiter", type=int, default=3000)
    ap.add_argument("--keep-temp", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    # Config
    a_lo,a_hi = [float(x) for x in args.alpha_bounds.split(",")]
    l_lo,l_hi = [float(x) for x in args.lambda_bounds.split(",")]
    ml_mu, ml_sig = [float(x) for x in args.ml_prior.split(",")]
    yd_lo, yd_hi = [float(x) for x in args.ml_bounds.split(",")]
    cfg = {
        "alpha_bounds": (a_lo,a_hi), "lambda_bounds": (l_lo,l_hi),
        "alpha0": args.alpha0, "lam0": args.lam0,
        "fit_ml": bool(args.fit_ml), "y_disk_init": 1.0, "y_bul_init": 1.0,
        "yd_bounds": (yd_lo, yd_hi), "yb_bounds": (yd_lo, yd_hi),
        "fit_errfloor": bool(args.fit_errfloor), "sint_init": float(args.sint_init), "sint_max": float(args.sint_max),
        "robust": bool(args.robust), "huber_delta": float(args.huber_delta),
        "maxiter": int(args.maxiter),
        "priors": {"alpha_mu": args.alpha_mu, "alpha_sig": args.alpha_sig,
                   "lam_mu": args.lam_mu, "lam_sig": args.lam_sig,
                   "ml_mu": ml_mu, "ml_sig": ml_sig,
                   "sint_mu": 0.0, "sint_sig": args.sint_max/3.0}
    }

    # I/O
    os.makedirs(args.out, exist_ok=True)
    conv_dir = os.path.join(args.out, "converted"); os.makedirs(conv_dir, exist_ok=True)
    plot_dir = os.path.join(args.out, "plots"); os.makedirs(plot_dir, exist_ok=True)

    # Source
    data_dir = None; tmp_root = None
    try:
        if args.zip:
            data_dir, tmp_root = extract_zip_to_temp(args.zip)
        else:
            data_dir = args.data_dir

        files = sorted([f for f in os.listdir(data_dir) if re.search(r"\.(dat|txt|csv)$", f, re.I)])
        rows = []
        for fn in files:
            path = os.path.join(data_dir, fn); tag = os.path.splitext(fn)[0]
            try:
                df = parse_rotmod_to_standard(path)
                df.to_csv(os.path.join(conv_dir, f"{tag}.csv"), index=False)
                fit = fit_one(df, cfg)
                outrow = {"file": fn, **fit}
                rows.append(outrow)
                if args.make_plots:
                    save_plot(df, tag, args.out, fit)
            except Exception as e:
                rows.append({"file": fn, "error": str(e)})
        res = pd.DataFrame(rows)
        res.to_csv(os.path.join(args.out, "results.csv"), index=False)
        ok = res.dropna(subset=["alpha_eff","lambda_kpc"]) if "alpha_eff" in res.columns else pd.DataFrame()
        summary = {
            "num_total": int(res.shape[0]),
            "num_success": int(ok.shape[0]),
            "median_alpha": float(np.nanmedian(ok["alpha_eff"])) if ok.shape[0] else None,
            "median_lambda": float(np.nanmedian(ok["lambda_kpc"])) if ok.shape[0] else None,
            "median_chi2_red": float(np.nanmedian(ok["chi2_red"])) if "chi2_red" in ok.columns else None
        }
        print(json.dumps(summary, indent=2))
    finally:
        if args.zip and tmp_root and (not args.keep_temp):
            shutil.rmtree(tmp_root, ignore_errors=True)

if __name__ == "__main__":
    main()
