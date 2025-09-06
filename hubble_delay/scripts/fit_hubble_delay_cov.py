#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_hubble_delay_cov.py  (compact, covariance-aware)
- SN full covariance (--sn-cov), optional intrinsic scatter (mag)
- BAO DV/rd covariance (--bao-cov)
- BAO [DM/rd, Hz*rd] stacked covariance (--bao2-cov-block)
- CPL background; optional EDE-like bump (cpl-ede)
- array-safe distances (np.atleast_1d)

Outputs: results.json, sn_fit.png (if SN), bao_fit.png (if BAO)
"""
import os, argparse, json, numpy as np, pandas as pd, matplotlib.pyplot as plt

c_kms = 299792.458

def E_cpl(z, Om, Or, Ok, w0, wa, ede=None):
    z = np.atleast_1d(z).astype(float)
    a = 1.0/(1.0+z)
    X = (1.0 - Om - Or - Ok)
    fac = a**(-3.0*(1.0 + w0 + wa)) * np.exp(3.0*wa*(a - 1.0))
    rhoX = X * fac
    rho_e = 0.0
    if ede is not None:
        fe, zc, sigma = ede["f_e"], ede["z_c"], ede["sigma"]
        ln = (np.log((1.0+z)/(1.0+zc))) / sigma
        S = fe * np.exp(-0.5 * ln**2)
        S0 = fe * np.exp(-0.5 * (np.log(1.0/(1.0+zc))/sigma)**2)
        rho_e = np.maximum(S - S0, 0.0)
    return np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + Ok*(1+z)**2 + rhoX + rho_e)

def _trapz_invE(z, Ez):
    return np.trapz(1.0/np.maximum(Ez, 1e-12), z)

def comoving_distance(z, H0, Om, Or, Ok, w0, wa, ede=None, nz=1024):
    z = np.atleast_1d(z).astype(float)
    DM = np.zeros_like(z)
    for i, zi in enumerate(z):
        zz = np.linspace(0.0, zi, max(nz, 16))
        Ez = E_cpl(zz, Om, Or, Ok, w0, wa, ede)
        chi = (c_kms/H0) * _trapz_invE(zz, Ez)
        if Ok > 0:
            s = np.sqrt(Ok); DM[i] = (c_kms/H0)/s * np.sinh(s * chi * H0 / c_kms)
        elif Ok < 0:
            s = np.sqrt(-Ok); DM[i] = (c_kms/H0)/s * np.sin(s * chi * H0 / c_kms)
        else:
            DM[i] = chi
    return DM if DM.size>1 else float(DM[0])

def distances(z, H0, Om, Or, Ok, w0, wa, ede=None):
    z = np.atleast_1d(z).astype(float)
    DM = comoving_distance(z, H0, Om, Or, Ok, w0, wa, ede)
    DA = DM/(1.0+z); DL = (1.0+z)*DM
    return DM, DA, DL

def DV(z, H0, Om, Or, Ok, w0, wa, ede=None):
    z = np.atleast_1d(z).astype(float)
    Ez = E_cpl(z, Om, Or, Ok, w0, wa, ede)
    DM, DA, _ = distances(z, H0, Om, Or, Ok, w0, wa, ede)
    DVv = (( (1.0+z)**2 * DA**2 * (c_kms*z)/(H0*np.maximum(Ez,1e-12)) ))**(1.0/3.0)
    return DVv if DVv.size>1 else float(DVv[0])

def mu_model(z, H0, Om, Or, Ok, w0, wa, M, ede=None):
    z = np.atleast_1d(z).astype(float)
    _, _, DL = distances(z, H0, Om, Or, Ok, w0, wa, ede)
    mu = 5.0*np.log10(np.maximum(DL,1e-30)*1e6) - 5.0 + M
    return mu if mu.size>1 else float(mu[0])

def load_cov(path, n_expected=None, jitter=0.0):
    try: arr = pd.read_csv(path, header=None).values
    except Exception: arr = np.loadtxt(path)
    arr = np.array(arr, float)
    if arr.ndim==1:
        if n_expected is None: raise ValueError("Triangular cov needs n_expected")
        n = int((np.sqrt(1+8*arr.size)-1)//2)
        if n*(n+1)//2 != arr.size: raise ValueError("Bad triangular length")
        C = np.zeros((n,n)); idx = np.tril_indices(n); C[idx]=arr; C = C + C.T - np.diag(np.diag(C))
    else:
        if arr.shape[0]!=arr.shape[1]: raise ValueError("Cov not square")
        C = arr
    if n_expected is not None and C.shape!=(n_expected,n_expected): raise ValueError("Shape mismatch")
    if jitter>0: C = C + np.eye(C.shape[0])*jitter
    return 0.5*(C+C.T)

def chi2_gaussian(res, C):
    C = 0.5*(C + C.T)
    d = np.diag(C)
    eps = 1e-12*np.median(d[d>0]) if np.any(d>0) else 1e-8
    L = np.linalg.cholesky(C + np.eye(C.shape[0])*eps)
    y = np.linalg.solve(L, res)
    return float(y @ y)

def chi2_sn(df, p, ede=None, cov=None, s_int_mag=0.0):
    z = df["z"].values; mu_obs = df["mu"].values
    sig = df["sigma_mu"].values if "sigma_mu" in df.columns else None
    mu_th = mu_model(z, p["H0"], p["Om"], p.get("Or",0.0), p["Ok"], p["w0"], p["wa"], p["M"], ede)
    r = (mu_obs - mu_th).astype(float)
    if cov is not None:
        C = cov.copy()
        if s_int_mag>0: C[np.diag_indices_from(C)] += s_int_mag**2
        return chi2_gaussian(r, C)
    if sig is None: raise ValueError("Need sigma_mu or SN cov")
    s2 = sig**2 + (s_int_mag**2 if s_int_mag>0 else 0.0)
    return float(np.sum((r**2)/np.maximum(s2,1e-20)))

def chi2_bao_A(df, p, rd, ede=None, cov=None):
    z = df["z"].values; obs = df["DV_over_rd"].values
    sig = df["sigma_DV_over_rd"].values if "sigma_DV_over_rd" in df.columns else None
    th = DV(z, p["H0"], p["Om"], p.get("Or",0.0), p["Ok"], p["w0"], p["wa"], ede)/rd
    r = (obs - th).astype(float)
    if cov is not None: return chi2_gaussian(r, cov)
    if sig is None: raise ValueError("Need BAO(DV) sigmas or cov")
    return float(np.sum((r**2)/np.maximum(sig**2,1e-20)))

def chi2_bao_B(df, p, rd, ede=None, cov_block=None):
    z = df["z"].values
    vec_obs, vec_th = [], []
    if "DM_over_rd" in df.columns:
        DM_th = distances(z, p["H0"], p["Om"], p.get("Or",0.0), p["Ok"], p["w0"], p["wa"], ede)[0]/rd
        vec_th += DM_th.tolist(); vec_obs += df["DM_over_rd"].values.tolist()
    if "Hz_rd" in df.columns:
        Ez = E_cpl(z, p["Om"], p.get("Or",0.0), p["Ok"], p["w0"], p["wa"], ede)
        Hzrd_th = (p["H0"]*Ez)*rd/c_kms
        vec_th += Hzrd_th.tolist(); vec_obs += df["Hz_rd"].values.tolist()
    r = np.array(vec_obs) - np.array(vec_th)
    if cov_block is not None: return chi2_gaussian(r, cov_block)
    # diagonal fallback
    sig2 = []
    if "DM_over_rd" in df.columns and "sigma_DM_over_rd" in df.columns:
        sig2 += (df["sigma_DM_over_rd"].values**2).tolist()
    if "Hz_rd" in df.columns and "sigma_Hz_rd" in df.columns:
        sig2 += (df["sigma_Hz_rd"].values**2).tolist()
    if len(sig2)!=len(r): raise ValueError("Need BAO2 cov or complete per-point sigmas")
    return float(np.sum((r**2)/np.maximum(np.array(sig2),1e-20)))

def run(args):
    SN = pd.read_csv(args.sn) if args.sn else None
    BAO = pd.read_csv(args.bao) if args.bao else None
    BAO2 = pd.read_csv(args.bao2) if args.bao2 else None

    SN_COV = load_cov(args.sn_cov, n_expected=len(SN), jitter=args.sn_cov_jitter) if args.sn_cov else None
    BAO_COV = load_cov(args.bao_cov, n_expected=len(BAO), jitter=args.bao_cov_jitter) if args.bao_cov else None
    BAO2_COV = None
    if args.bao2_cov_block:
        n = 0
        if BAO2 is not None:
            if "DM_over_rd" in BAO2.columns: n += len(BAO2)
            if "Hz_rd" in BAO2.columns: n += len(BAO2)
        BAO2_COV = load_cov(args.bao2_cov_block, n_expected=n, jitter=args.bao2_cov_jitter)

    ede = None
    if args.model=="cpl-ede":
        fe, zc, sig = map(float, args.ede_params.split(","))
        ede = {"f_e":fe, "z_c":zc, "sigma":sig}

    bounds = {"H0":tuple(map(float,args.H0_bounds.split(","))),
              "Om":(0.0,1.0), "Ok":(0.0,0.0), "w0":(-1.0,-1.0), "wa":(0.0,0.0),
              "M":(-5.0,5.0)}
    rd = 147.0  # Fixed rd for LCDM model
    free_rd = False  # Fixed rd for LCDM model
    rd_prior = (args.rd_mu, args.rd_sig) if (args.rd_mu is not None and args.rd_sig is not None) else None
    H0_prior = (args.H0_prior_mu, args.H0_prior_sig) if (args.H0_prior_mu is not None and args.H0_prior_sig is not None) else None

    def chi2_total(vec):
        H0, Om, M = vec[:3]  # Only optimize H0, Om, M
        Ok, w0, wa = 0.0, -1.0, 0.0  # Fixed parameters for LCDM
        if not(bounds["H0"][0]<=H0<=bounds["H0"][1] and 0<=Om<=1 and -5<=M<=5):
            return 1e20
        p = {"H0":H0,"Om":Om,"Ok":Ok,"w0":w0,"wa":wa,"M":M,"Or":0.0}
        val=0.0
        if SN is not None: val += chi2_sn(SN,p,ede,SN_COV,args.sn_intrinsic_mag)
        if BAO is not None: val += chi2_bao_A(BAO,p,rd,ede,BAO_COV)
        if BAO2 is not None: val += chi2_bao_B(BAO2,p,rd,ede,BAO2_COV)
        if H0_prior is not None:
            mu,sig = H0_prior; val += ((H0-mu)/sig)**2
        if free_rd and rd_prior is not None:
            mu,sig = rd_prior; val += ((vec[6]-mu)/sig)**2
        return float(val)

    x0 = np.array([args.H0_init,args.Om_init,args.M_init],float)  # Only H0, Om, M

    try:
        from scipy.optimize import minimize
        res = minimize(chi2_total, x0=x0, method="Nelder-Mead",
                       options={"maxiter":args.maxiter,"xatol":1e-6,"fatol":1e-6})
        xbest, chi2 = res.x, float(res.fun)
    except Exception:
        # fallback random search
        step = np.array([2.0,0.05,0.5], float)  # Only H0, Om, M
        x = x0.copy(); best = chi2_total(x)
        for _ in range(args.maxiter):
            cand = x + (np.random.rand(x.size)-0.5)*step
            val = chi2_total(cand)
            if val<best: x,best=cand,val; step*=0.98
            else: step*=0.999
        xbest, chi2 = x, best

    H0, Om, M = map(float, xbest[:3])
    Ok, w0, wa = 0.0, -1.0, 0.0  # Fixed parameters
    out = {"H0":H0,"Om":Om,"Ok":Ok,"w0":w0,"wa":wa,"M":M,
           "chi2":chi2,"model":args.model,"free_rd":free_rd,
           "rd": rd,  # Fixed rd value
           "used_sn_cov": bool(SN_COV is not None),
           "used_bao_cov": bool(BAO_COV is not None),
           "used_bao2_cov_block": bool(BAO2_COV is not None),
           "sn_intrinsic_mag": args.sn_intrinsic_mag}

    # DOF
    npts=0
    if SN is not None: npts += len(SN)
    if BAO is not None: npts += len(BAO)
    if BAO2 is not None:
        m = 0
        if "DM_over_rd" in BAO2.columns: m += len(BAO2)
        if "Hz_rd" in BAO2.columns: m += len(BAO2)
        npts += m
    npar = 3  # Only H0, Om, M are free parameters
    out["dof"] = max(npts-npar,1); out["chi2_red"]= out["chi2"]/out["dof"] if out["dof"] else None

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out,"results.json"),"w") as f: json.dump(out,f,indent=2)

    # Quick plots
    if SN is not None:
        z = np.linspace(0, float(np.nanmax(SN["z"]))*1.05, 400)
        ede_dict=None
        if args.model=="cpl-ede":
            fe,zc,sig = map(float,args.ede_params.split(","))
            ede_dict={"f_e":fe,"z_c":zc,"sigma":sig}
        mu_th = mu_model(z,H0,Om,0.0,Ok,w0,wa,M,ede_dict)
        plt.figure(figsize=(6,4))
        yerr = SN["sigma_mu"] if "sigma_mu" in SN.columns else None
        plt.errorbar(SN["z"],SN["mu"],yerr=yerr,fmt='.',label="SN")
        plt.plot(z,mu_th,label="Best-fit")
        plt.xlabel("z"); plt.ylabel("Distance modulus μ"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.out,"sn_fit.png")); plt.close()

    if BAO is not None:
        z = BAO["z"].values
        ede_dict=None
        if args.model=="cpl-ede":
            fe,zc,sig = map(float,args.ede_params.split(","))
            ede_dict={"f_e":fe,"z_c":zc,"sigma":sig}
        dv_th = DV(z,H0,Om,0.0,Ok,w0,wa,ede_dict)/out["rd"]
        plt.figure(figsize=(6,4))
        plt.errorbar(z,BAO["DV_over_rd"],yerr=BAO["sigma_DV_over_rd"],fmt='o',label="BAO DV/rd")
        plt.plot(z,dv_th,label="Model")
        plt.xlabel("z"); plt.ylabel("DV/rd"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.out,"bao_fit.png")); plt.close()

    print(json.dumps(out,indent=2))

def parse():
    ap = argparse.ArgumentParser(description="Covariance-aware H0 fitter (CPL / CPL-EDE).")
    ap.add_argument("--sn", type=str); ap.add_argument("--sn-cov", type=str, dest="sn_cov", default=None)
    ap.add_argument("--sn-intrinsic-mag", type=float, default=0.0); ap.add_argument("--sn-cov-jitter", type=float, default=1e-8)
    ap.add_argument("--bao", type=str); ap.add_argument("--bao-cov", type=str, default=None); ap.add_argument("--bao-cov-jitter", type=float, default=1e-8)
    ap.add_argument("--bao2", type=str); ap.add_argument("--bao2-cov-block", type=str, default=None); ap.add_argument("--bao2-cov-jitter", type=float, default=1e-8)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--model", type=str, default="cpl", choices=["cpl","cpl-ede"]); ap.add_argument("--ede-params", type=str, default="0.05,3000,0.5")
    ap.add_argument("--H0-init", type=float, default=70.0); ap.add_argument("--Om-init", type=float, default=0.3)
    ap.add_argument("--Ok-init", type=float, default=0.0); ap.add_argument("--w0-init", type=float, default=-1.0); ap.add_argument("--wa-init", type=float, default=0.0)
    ap.add_argument("--M-init", type=float, default=0.0)
    ap.add_argument("--H0-bounds", type=str, default="60,85")
    ap.add_argument("--rd", type=float, default=None); ap.add_argument("--free-rd", action="store_true")
    ap.add_argument("--rd-init", type=float, default=147.0); ap.add_argument("--rd-prior", type=str, default=None)
    ap.add_argument("--H0-prior", type=str, default=None)
    ap.add_argument("--maxiter", type=int, default=5000)
    args = ap.parse_args()
    # parse priors
    args.rd_mu=args.rd_sig=None
    if args.rd_prior:
        mu,sig = args.rd_prior.split(","); args.rd_mu=float(mu); args.rd_sig=float(sig)
    args.H0_prior_mu=args.H0_prior_sig=None
    if args.H0_prior:
        mu,sig = args.H0_prior.split(","); args.H0_prior_mu=float(mu); args.H0_prior_sig=float(sig)
    return args

if __name__=="__main__":
    run(parse())
