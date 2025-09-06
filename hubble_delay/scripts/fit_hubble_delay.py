#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_hubble_delay.py
-------------------
Fits late-time expansion (and optionally an early-time 'delay bump') to SN/BAO/H(z)
to study the H0 tension within a delay/CPL background.

Datasets (CSV expected):
  - Supernovae: columns z, mu, sigma_mu
  - BAO (option A): columns z, DV_over_rd, sigma_DV_over_rd
  - BAO (option B): columns z, DM_over_rd, sigma_DM_over_rd, Hz_rd, sigma_Hz_rd
  - Cosmic Chronometers H(z): columns z, Hz, sigma_Hz
  - Optional H0 prior: value and sigma via CLI

Model:
  E^2(z) = Ω_m(1+z)^3 + Ω_r(1+z)^4 + Ω_k(1+z)^2
           + Ω_X * exp[ 3∫_0^z (1+w(z'))/(1+z') dz' ] + Ω_EDE(z)
  with CPL: w(z) = w0 + wa z/(1+z); Ω_X = 1-Ω_m-Ω_r-Ω_k.
  Ω_EDE(z) is optional 'delay bump': f_e * exp[-(ln(1+z/zc))^2/(2σ^2)] normalized to vanish at z=0.
  rd can be fixed or treated as a free nuisance with Gaussian prior.

Usage examples:
  python fit_hubble_delay.py --sn pantheon.csv --bao bao.csv --cc cc.csv \
      --out ./h0_out --model cpl --H0-prior 73.0,1.0 --lambda-bounds 1,50

  python fit_hubble_delay.py --sn pantheon.csv --bao2 bao_dm_h.csv --free-rd \
      --out ./h0_out --model cpl-ede --ede-params 0.05,3000,0.5
"""
import os, argparse, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

c_kms = 299792.458

def E_cpl(z, Om, Or, Ok, w0, wa, ede=None):
    """E(z) for CPL with optional EDE-like bump (ede dict or None)."""
    a = 1.0/(1.0+z)
    # CPL factor
    w = lambda a: w0 + wa*(1.0 - a)
    # Integral 3 ∫ (1+w) da/a => analytic for CPL: a^{-3(1+w0+wa)} * exp(3 wa(a-1))
    X = (1.0 - Om - Or - Ok)
    fac = a**(-3.0*(1.0 + w0 + wa)) * np.exp(3.0*wa*(a - 1.0))
    rhoX = X * fac

    rho_e = 0.0
    if ede is not None:
        # log-normal bump in (1+z), normalized so that rho_e(z=0)=0
        fe, zc, sigma = ede["f_e"], ede["z_c"], ede["sigma"]
        # shape S(z) with peak near zc
        ln_term = (np.log((1.0+z)/(1.0+zc))) / sigma
        S = fe * np.exp(-0.5 * ln_term**2)
        # enforce S(0)=0 by subtraction of S0, and keep positive
        S0 = fe * np.exp(-0.5 * (np.log(1.0/(1.0+zc))/sigma)**2)
        rho_e = np.maximum(S - S0, 0.0)

    return np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + Ok*(1+z)**2 + rhoX + rho_e)

def comoving_distance(z, H0, Om, Or, Ok, w0, wa, ede=None, nz=2048):
    z_array = np.atleast_1d(z)
    results = []
    for zi in z_array:
        zz = np.linspace(0, zi, nz)
        Ez = E_cpl(zz, Om, Or, Ok, w0, wa, ede)
        chi = (c_kms/H0) * np.trapz(1.0/Ez, zz)
        # curvature
        sqrtOk = np.sqrt(np.abs(Ok))
        if Ok > 0:
            DM = (c_kms/H0)/sqrtOk * np.sinh(sqrtOk * H0*chi/c_kms)
        elif Ok < 0:
            DM = (c_kms/H0)/sqrtOk * np.sin(sqrtOk * H0*chi/c_kms)
        else:
            DM = chi
        results.append(DM)
    return np.array(results)

def distances(z, H0, Om, Or, Ok, w0, wa, ede=None):
    DM = comoving_distance(z, H0, Om, Or, Ok, w0, wa, ede)
    z_array = np.atleast_1d(z)
    DA = DM/(1+z_array)
    DL = (1+z_array)*DM
    return DM, DA, DL

def DV(z, H0, Om, Or, Ok, w0, wa, ede=None):
    Ez = E_cpl(z, Om, Or, Ok, w0, wa, ede)
    DM, DA, DL = distances(z, H0, Om, Or, Ok, w0, wa, ede)
    return ( ( (1+z)**2 * DA**2 * (c_kms*z)/ (H0*Ez) ) )**(1.0/3.0)

def mu_model(z, H0, Om, Or, Ok, w0, wa, M, ede=None):
    _, _, DL = distances(z, H0, Om, Or, Ok, w0, wa, ede)
    # DL is in Mpc if H0 in km/s/Mpc and c in km/s
    mu = 5.0*np.log10(DL*1e6) - 5.0 + M  # 10 pc = 1e-5 Mpc → +25; but absorb into M
    return mu

def chi2_sn(df, pars, ede=None):
    z = df["z"].values
    mu_obs = df["mu"].values
    sig = df["sigma_mu"].values
    H0, Om, Ok, w0, wa, M = pars["H0"], pars["Om"], pars["Ok"], pars["w0"], pars["wa"], pars["M"]
    Or = pars.get("Or", 0.0)
    mu_th = mu_model(z, H0, Om, Or, Ok, w0, wa, M, ede)
    chi = (mu_obs - mu_th)/np.maximum(sig, 1e-6)
    return float(np.sum(chi**2))

def chi2_bao_A(df, pars, rd, ede=None):
    # BAO option A: DV/rd
    z = df["z"].values
    dv_obs = df["DV_over_rd"].values
    sig = df["sigma_DV_over_rd"].values
    H0, Om, Ok, w0, wa = pars["H0"], pars["Om"], pars["Ok"], pars["w0"], pars["wa"]
    Or = pars.get("Or", 0.0)
    dv_th = np.array([ DV(zi, H0, Om, Or, Ok, w0, wa, ede)/rd for zi in z ])
    chi = (dv_obs - dv_th)/np.maximum(sig, 1e-6)
    return float(np.sum(chi**2))

def chi2_bao_B(df, pars, rd, ede=None):
    # BAO option B: DM/rd and H(z)*rd
    z = df["z"].values
    cols = df.columns
    H0, Om, Ok, w0, wa = pars["H0"], pars["Om"], pars["Ok"], pars["w0"], pars["wa"]
    Or = pars.get("Or", 0.0)
    chi2 = 0.0
    if "DM_over_rd" in cols:
        DM_th = np.array([ distances(zi, H0, Om, Or, Ok, w0, wa, ede)[0]/rd for zi in z ])
        chi2 += np.sum(((df["DM_over_rd"].values - DM_th)/df["sigma_DM_over_rd"].values)**2)
    if "Hz_rd" in cols:
        Ez = E_cpl(z, Om, Or, Ok, w0, wa, ede)
        Hzrd_th = (H0*Ez)*rd/c_kms  # dimensionless
        chi2 += np.sum(((df["Hz_rd"].values - Hzrd_th)/df["sigma_Hz_rd"].values)**2)
    return float(chi2)

def chi2_cc(df, pars, ede=None):
    z = df["z"].values
    H_obs = df["Hz"].values
    sig = df["sigma_Hz"].values
    H0, Om, Ok, w0, wa = pars["H0"], pars["Om"], pars["Ok"], pars["w0"], pars["wa"]
    Or = pars.get("Or", 0.0)
    Ez = E_cpl(z, Om, Or, Ok, w0, wa, ede)
    H_th = H0*Ez
    chi = (H_obs - H_th)/np.maximum(sig, 1e-6)
    return float(np.sum(chi**2))

def run_fit(args):
    # Load datasets if provided
    SN = pd.read_csv(args.sn) if args.sn else None
    BAO = pd.read_csv(args.bao) if args.bao else None
    BAO2 = pd.read_csv(args.bao2) if args.bao2 else None
    CC = pd.read_csv(args.cc) if args.cc else None

    # Params and priors
    pars = {
        "H0": args.H0_init, "Om": args.Om_init, "Ok": args.Ok_init,
        "w0": args.w0_init, "wa": args.wa_init, "M": args.M_init, "Or": args.Or_init
    }
    bounds = {
        "H0": tuple(map(float, args.H0_bounds.split(","))),
        "Om": (0.0, 1.0), "Ok": (-0.1, 0.1),
        "w0": (-2.0, 0.0), "wa": (-2.0, 2.0),
        "M": (-5.0, 5.0), "Or": (0.0, 0.01)
    }
    # rd prior or free
    rd = args.rd if (args.rd is not None) else args.rd_init
    free_rd = bool(args.free_rd)
    rd_prior = (args.rd_mu, args.rd_sig) if (args.rd_mu is not None and args.rd_sig is not None) else None

    # EDE-like bump
    ede = None
    if args.model == "cpl-ede":
        fe, zc, sig = map(float, args.ede_params.split(","))
        ede = {"f_e": fe, "z_c": zc, "sigma": sig}

    # H0 prior
    H0_prior = (args.H0_prior_mu, args.H0_prior_sig) if (args.H0_prior_mu is not None and args.H0_prior_sig is not None) else None

    # Objective function
    def chi2_total(vec):
        H0, Om, Ok, w0, wa, M = vec[:6]
        Or = pars["Or"]
        p = {"H0":H0, "Om":Om, "Ok":Ok, "w0":w0, "wa":wa, "M":M, "Or":Or}
        # bounds check
        if not (bounds["H0"][0] <= H0 <= bounds["H0"][1]): return 1e20
        if not (bounds["Om"][0] <= Om <= bounds["Om"][1]): return 1e20
        if not (bounds["Ok"][0] <= Ok <= bounds["Ok"][1]): return 1e20
        if not (bounds["w0"][0] <= w0 <= bounds["w0"][1]): return 1e20
        if not (bounds["wa"][0] <= wa <= bounds["wa"][1]): return 1e20
        if not (bounds["M"][0] <= M <= bounds["M"][1]): return 1e20

        val = 0.0
        if SN is not None: val += chi2_sn(SN, p, ede)
        if BAO is not None: val += chi2_bao_A(BAO, p, rd, ede)
        if BAO2 is not None: val += chi2_bao_B(BAO2, p, rd, ede)
        if CC is not None: val += chi2_cc(CC, p, ede)
        if H0_prior is not None:
            mu, sig = H0_prior
            val += ((H0 - mu)/sig)**2
        if free_rd and rd_prior is not None:
            mu, sig = rd_prior
            val += ((vec[6] - mu)/sig)**2
        return float(val)

    # Initial vector
    x0 = np.array([pars["H0"], pars["Om"], pars["Ok"], pars["w0"], pars["wa"], pars["M"]], dtype=float)
    if free_rd:
        x0 = np.append(x0, rd)

    # Minimize
    try:
        from scipy.optimize import minimize
        res = minimize(chi2_total, x0=x0, method="Nelder-Mead",
                       options={"maxiter": args.maxiter, "xatol": 1e-6, "fatol": 1e-6})
        xbest = res.x; chi2 = float(res.fun); success = bool(res.success)
    except Exception:
        # fallback: random search
        x = x0.copy(); step = np.array([2.0, 0.05, 0.01, 0.2, 0.2, 0.5] + ([2.0] if free_rd else []), dtype=float)
        best = chi2_total(x); success=True
        for _ in range(args.maxiter):
            cand = x + (np.random.rand(x.size)-0.5)*step
            val = chi2_total(cand)
            if val < best:
                x, best = cand, val; step *= 0.98
            else:
                step *= 0.999
        xbest, chi2 = x, best

    # Unpack
    H0, Om, Ok, w0, wa, M = map(float, xbest[:6])
    out = {"H0":H0, "Om":Om, "Ok":Ok, "w0":w0, "wa":wa, "M":M,
           "chi2": chi2, "dof": None, "model": args.model,
           "free_rd": free_rd, "rd": float(xbest[6]) if free_rd else (args.rd if args.rd is not None else args.rd_init)}
    # DOF rough count
    npts = 0
    if SN is not None: npts += len(SN)
    if BAO is not None: npts += len(BAO)
    if BAO2 is not None: npts += len(BAO2)
    if CC is not None: npts += len(CC)
    npar = 6 + (1 if free_rd else 0)
    out["dof"] = max(npts - npar, 1)
    out["chi2_red"] = out["chi2"]/out["dof"] if out["dof"] else None

    # Save results
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Make quick diagnostic plot if SN provided
    if SN is not None:
        z = np.linspace(0, SN["z"].max()*1.05, 400)
        ede_dict = None
        if args.model == "cpl-ede":
            fe, zc, sig = map(float, args.ede_params.split(","))
            ede_dict = {"f_e": fe, "z_c": zc, "sigma": sig}
        mu_th = mu_model(z, H0, Om, 0.0, Ok, w0, wa, M, ede_dict)
        plt.figure(figsize=(6,4))
        plt.errorbar(SN["z"], SN["mu"], yerr=SN["sigma_mu"], fmt='.', label="SN")
        plt.plot(z, mu_th, label="Best-fit model")
        plt.xlabel("z"); plt.ylabel("Distance modulus μ")
        plt.title(f"H0={H0:.1f}, Om={Om:.3f}, w0={w0:.2f}, wa={wa:.2f}")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.out, "sn_fit.png")); plt.close()

    # BAO plot if BAO
    if BAO is not None:
        z = BAO["z"].values
        ede_dict = None
        if args.model == "cpl-ede":
            fe, zc, sig = map(float, args.ede_params.split(","))
            ede_dict = {"f_e": fe, "z_c": zc, "sigma": sig}
        dv_th = np.array([ DV(zi, H0, Om, 0.0, Ok, w0, wa, ede_dict)/out["rd"] for zi in z ])
        plt.figure(figsize=(6,4))
        plt.errorbar(z, BAO["DV_over_rd"], yerr=BAO["sigma_DV_over_rd"], fmt='o', label="BAO DV/rd")
        plt.plot(z, dv_th, label="Model")
        plt.xlabel("z"); plt.ylabel("DV/rd"); plt.tight_layout()
        plt.legend(); plt.savefig(os.path.join(args.out, "bao_fit.png")); plt.close()

    print(json.dumps(out, indent=2))

def parse_args():
    ap = argparse.ArgumentParser(description="Fit H0 with delay/CPL background to SN/BAO/CC.")
    ap.add_argument("--sn", type=str, help="CSV with columns z,mu,sigma_mu")
    ap.add_argument("--bao", type=str, help="CSV with columns z,DV_over_rd,sigma_DV_over_rd")
    ap.add_argument("--bao2", type=str, help="CSV with columns z, [DM_over_rd, sigma_DM_over_rd, Hz_rd, sigma_Hz_rd]")
    ap.add_argument("--cc", type=str, help="CSV with columns z,Hz,sigma_Hz")
    ap.add_argument("--out", type=str, required=True, help="Output directory")
    ap.add_argument("--model", type=str, default="cpl", choices=["cpl","cpl-ede"])
    ap.add_argument("--ede-params", type=str, default="0.05,3000,0.5", help="f_e,z_c,sigma for EDE-like bump")
    # Initials and bounds
    ap.add_argument("--H0-init", dest="H0_init", type=float, default=70.0)
    ap.add_argument("--Om-init", dest="Om_init", type=float, default=0.3)
    ap.add_argument("--Ok-init", dest="Ok_init", type=float, default=0.0)
    ap.add_argument("--w0-init", dest="w0_init", type=float, default=-1.0)
    ap.add_argument("--wa-init", dest="wa_init", type=float, default=0.0)
    ap.add_argument("--M-init", dest="M_init", type=float, default=0.0)
    ap.add_argument("--Or-init", dest="Or_init", type=float, default=0.0)
    ap.add_argument("--H0-bounds", type=str, default="60,85")
    # rd handling
    ap.add_argument("--rd", type=float, default=None, help="Fix rd to this value (Mpc)")
    ap.add_argument("--free-rd", action="store_true", help="Treat rd as free parameter")
    ap.add_argument("--rd-init", dest="rd_init", type=float, default=147.0)
    ap.add_argument("--rd-prior", type=str, default=None, help="mu,sig for rd prior (Mpc)")
    # H0 prior
    ap.add_argument("--H0-prior", type=str, default=None, help="mu,sig for H0 prior (km/s/Mpc)")
    ap.add_argument("--maxiter", type=int, default=4000)
    args = ap.parse_args()

    # Parse priors
    args.rd_mu, args.rd_sig = (None, None)
    if args.rd_prior:
        mu, sig = args.rd_prior.split(",")
        args.rd_mu, args.rd_sig = float(mu), float(sig)
    args.H0_prior_mu, args.H0_prior_sig = (None, None)
    if args.H0_prior:
        mu, sig = args.H0_prior.split(",")
        args.H0_prior_mu, args.H0_prior_sig = float(mu), float(sig)

    return args

if __name__ == "__main__":
    run_fit(parse_args())
