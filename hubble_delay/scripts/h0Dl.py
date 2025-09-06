#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Downloader for H0-tension datasets:
 - Pantheon+SH0ES (supernovae distances)
 - SDSS DR17 BAO (table scrape + zipped measurement bundles)
 - DESI DR2 BAO (paper PDF for now; switch to CSV when officially released)

Usage:
  python download_h0_data.py --all
  python download_h0_data.py --sn
  python download_h0_data.py --sdss
  python download_h0_data.py --desi

Outputs (default under ./data/):
  data/pantheon_plus/Pantheon+SH0ES.dat
  data/pantheon_plus/PantheonPlus_clean.csv   (اگر شد، به CSV هم تبدیل می‌کند)
  data/sdss_bao/DR17_table.csv
  data/sdss_bao/MultiTracer_CF_BAORSD_measurements.zip
  data/sdss_bao/MultiTracer_PK_BAORSD_measurements.zip
  data/desi_dr2/DESI_DR2_BAO_ResultsII_2503.14738.pdf
"""

import os
import re
import sys
import csv
import argparse
import time
from pathlib import Path

try:
    import requests
except Exception as e:
    print("ERROR: 'requests' is required. Install with: pip install requests")
    raise

# Optional deps:
HAS_PANDAS = True
HAS_BS4 = True
try:
    import pandas as pd
except Exception:
    HAS_PANDAS = False
try:
    from bs4 import BeautifulSoup
except Exception:
    HAS_BS4 = False


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

URLS = {
    # Pantheon+SH0ES GitHub raw (plus signs must be URL-encoded %2B)
    "pantheon_plus_raw": (
        "https://raw.githubusercontent.com/"
        "PantheonPlusSH0ES/DataRelease/main/"
        "Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
    ),
    # SDSS DR17 table (HTML page we will parse)
    "sdss_dr17_table": "https://www.sdss4.org/science/final-bao-and-rsd-measurements-table/",
    # SDSS multitracer bundles (zip)
    "sdss_cf_zip": "https://www.sdss4.org/wp-content/uploads/2020/07/MultiTracer_CF_BAORSD_measurements.zip",
    "sdss_pk_zip": "https://www.sdss4.org/wp-content/uploads/2020/07/MultiTracer_PK_BAORSD_measurements.zip",
    # DESI DR2 BAO results (PDF of the paper; numeric machine tables not public yet)
    "desi_dr2_pdf": "https://arxiv.org/pdf/2503.14738.pdf",
}

def _download(url: str, dest: Path, desc: str = "") -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"→ Downloading {desc or dest.name}\n  URL: {url}\n  → {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f:
            downloaded = 0
            t0 = time.time()
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total and downloaded % (1024 * 1024) == 0:
                        pct = 100 * downloaded / total
                        dt = time.time() - t0
                        rate = downloaded / (dt + 1e-9) / 1e6
                        print(f"    {downloaded/1e6:.1f} MB ({pct:.1f}%) @ {rate:.1f} MB/s", end="\r")
        tmp.rename(dest)
    print(f"✔ Saved: {dest}\n")
    return dest

def download_pantheon_plus(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    dat_path = outdir / "Pantheon+SH0ES.dat"
    _download(URLS["pantheon_plus_raw"], dat_path, "Pantheon+SH0ES.dat")

    # Try a light conversion to CSV (optional—it depends on columns present)
    # Many users just need z and distance modulus mu with uncertainty.
    # If columns 'z' and 'MU' (or 'mu') exist, we extract a minimal CSV.
    if HAS_PANDAS:
        try:
            # The .dat is usually whitespace-separated; comment lines start with '#'
            df = pd.read_csv(dat_path, comment="#", delim_whitespace=True)
            cols = [c.lower() for c in df.columns]
            # Heuristics for common column names in Pantheon+ tables:
            cand_z = None
            for key in ["z", "zcmb", "zhd", "redshift"]:
                if key in cols:
                    cand_z = df.columns[cols.index(key)]
                    break
            cand_mu = None
            for key in ["mu", "dmod", "distance_modulus"]:
                if key in cols:
                    cand_mu = df.columns[cols.index(key)]
                    break
            cand_sigma = None
            for key in ["muerr", "dmu", "sigma_mu", "sigma"]:
                if key in cols:
                    cand_sigma = df.columns[cols.index(key)]
                    break

            if cand_z and cand_mu:
                slim = df[[cand_z, cand_mu] + ([cand_sigma] if cand_sigma else [])].copy()
                slim.rename(
                    columns={
                        cand_z: "z",
                        cand_mu: "mu",
                        (cand_sigma or "dummy"): "sigma_mu",
                    },
                    inplace=True,
                )
                if "sigma_mu" not in slim.columns:
                    slim["sigma_mu"] = None
                slim.to_csv(outdir / "PantheonPlus_clean.csv", index=False)
                print(f"✔ Wrote CSV: {outdir / 'PantheonPlus_clean.csv'}")
            else:
                print("! Could not confidently locate (z, mu [, sigma_mu]) columns. Skipping CSV conversion.")
        except Exception as e:
            print(f"! CSV conversion skipped due to parse error: {e}")
    else:
        print("! pandas not installed; skipping CSV conversion for Pantheon+.")

def parse_sdss_dr17_table(html_text: str):
    """
    Parse the DR17 'Final BAO and RSD Measurements Table' page to extract
    the BAO-only DM/rd and DH/rd (and optionally DV/rd) lines with their numbers.
    """
    # Normalize whitespace
    txt = re.sub(r"\s+", " ", html_text)

    # Patterns for BAO-only lines (as displayed on the page)
    # Example fragments:
    # D_{M}(z)/r_{d} 10.23 +/- 0.17 13.36 +/- 0.21 ...
    # D_{H}(z)/r_{d} 25.00 +/- 0.76 22.33 +/- 0.58 ...
    patterns = {
        "DM_over_rd": r"D_\{M\}\(z\)/r_\{d\}\s+([0-9\.\-\+\s/]+?)D_\{H\}",
        "DH_over_rd": r"D_\{H\}\(z\)/r_\{d\}\s+([0-9\.\-\+\s/]+?)(?:Reference|RSD|BAO\+RSD|#|$)",
        "DV_over_rd": r"D_\{V\}\(z\)/r_\{d\}\s+([0-9\.\-\+\s/]+?)D_\{M\}",
    }

    out = {}
    for key, pat in patterns.items():
        m = re.search(pat, txt)
        if m:
            seq = m.group(1).strip()
            out[key] = seq

    # Effective redshifts are given as: "Effective Redshift 0.15 0.38 0.51 0.70 0.85 1.48 2.33 2.33"
    z_match = re.search(r"Effective Redshift\s+([0-9\.\s]+?)\s+Effective Volume", txt)
    z_values = []
    if z_match:
        z_values = [float(z) for z in z_match.group(1).split()]

    def parse_series(series_str):
        """
        Turn "10.23 +/- 0.17 13.36 +/- 0.21 ..." into list of (val, err)
        """
        toks = series_str.replace("+/-", "±").split()
        vals = []
        errs = []
        i = 0
        while i < len(toks):
            # expect: value, ±, error
            try:
                val = float(toks[i])
                if i + 2 < len(toks) and toks[i + 1] in ["±", "+/-"]:
                    err = float(toks[i + 2].replace("+", ""))
                    vals.append(val)
                    errs.append(err)
                    i += 3
                else:
                    # Some entries might be asymmetric like "18.33-0.62+0.57"
                    # Handle pattern X-A+B by taking mean error or store as text
                    asym = toks[i + 0]
                    m = re.match(r"([0-9\.]+)\-([0-9\.]+)\+([0-9\.]+)", asym)
                    if m:
                        val = float(m.group(1))
                        em = (float(m.group(2)) + float(m.group(3))) / 2.0
                        vals.append(val)
                        errs.append(em)
                        i += 1
                    else:
                        i += 1
            except Exception:
                i += 1
        return vals, errs

    rows = []
    if "DM_over_rd" in out and z_values:
        dm_vals, dm_errs = parse_series(out["DM_over_rd"])
        for k, z in enumerate(z_values[:len(dm_vals)]):
            rows.append({"z_eff": z, "quantity": "DM_over_rd", "value": dm_vals[k], "sigma": dm_errs[k]})
    if "DH_over_rd" in out and z_values:
        dh_vals, dh_errs = parse_series(out["DH_over_rd"])
        for k, z in enumerate(z_values[:len(dh_vals)]):
            rows.append({"z_eff": z, "quantity": "DH_over_rd", "value": dh_vals[k], "sigma": dh_errs[k]})
    if "DV_over_rd" in out and z_values:
        dv_vals, dv_errs = parse_series(out["DV_over_rd"])
        for k, z in enumerate(z_values[:len(dv_vals)]):
            rows.append({"z_eff": z, "quantity": "DV_over_rd", "value": dv_vals[k], "sigma": dv_errs[k]})

    return rows

def download_sdss_bao(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    # 1) Zip bundles
    _download(URLS["sdss_cf_zip"], outdir / "MultiTracer_CF_BAORSD_measurements.zip", "SDSS DR17 CF BAO+RSD zip")
    _download(URLS["sdss_pk_zip"], outdir / "MultiTracer_PK_BAORSD_measurements.zip", "SDSS DR17 PK BAO+RSD zip")

    # 2) Parse the DR17 table into a compact CSV
    table_csv = outdir / "DR17_table.csv"
    try:
        html = requests.get(URLS["sdss_dr17_table"], timeout=60).text
        if HAS_BS4 and not HAS_PANDAS:
            # Use BS4 + our regex parser
            rows = parse_sdss_dr17_table(html)
            if rows:
                with open(table_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["z_eff", "quantity", "value", "sigma"])
                    writer.writeheader()
                    for r in rows:
                        writer.writerow(r)
                print(f"✔ Wrote SDSS DR17 parsed table: {table_csv}")
            else:
                print("! Could not parse DR17 table; no rows extracted.")
        else:
            # If pandas is available, try read_html first; if fails, fall back to regex parser.
            saved = False
            if HAS_PANDAS:
                try:
                    tables = pd.read_html(html)
                    # Save all extracted tables to an Excel for inspection AND write a compact CSV later.
                    xls = outdir / "DR17_tables_raw.xlsx"
                    with pd.ExcelWriter(xls) as xlw:
                        for i, t in enumerate(tables):
                            t.to_excel(xlw, sheet_name=f"table_{i}", index=False)
                    print(f"✔ Saved raw HTML tables to: {xls}")
                    # Also build compact CSV via regex parser:
                    rows = parse_sdss_dr17_table(html)
                    if rows:
                        pd.DataFrame(rows).to_csv(table_csv, index=False)
                        print(f"✔ Wrote SDSS DR17 parsed table: {table_csv}")
                        saved = True
                except Exception:
                    pass
            if not saved:
                rows = parse_sdss_dr17_table(html)
                if rows:
                    with open(table_csv, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=["z_eff", "quantity", "value", "sigma"])
                        writer.writeheader()
                        for r in rows:
                            writer.writerow(r)
                    print(f"✔ Wrote SDSS DR17 parsed table: {table_csv}")
                else:
                    print("! Could not parse DR17 table; no rows extracted.")
    except Exception as e:
        print(f"! Failed to retrieve/parse SDSS DR17 table: {e}")

def download_desi_dr2(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    # Paper PDF
    _download(URLS["desi_dr2_pdf"], outdir / "DESI_DR2_BAO_ResultsII_2503.14738.pdf", "DESI DR2 BAO Results II (PDF)")
    print(
        "Note: DESI DR2 BAO numerical tables are typically distributed via collaboration pages "
        "or as likelihood packages. When a public CSV/JSON appears, update URLS and re-run."
    )

def main():
    ap = argparse.ArgumentParser(description="Downloader for H0-tension datasets")
    ap.add_argument("--all", action="store_true", help="Download all datasets")
    ap.add_argument("--sn", action="store_true", help="Download Pantheon+SH0ES")
    ap.add_argument("--sdss", action="store_true", help="Download SDSS DR17 BAO zips and parsed table")
    ap.add_argument("--desi", action="store_true", help="Download DESI DR2 BAO paper (PDF)")
    ap.add_argument("--out", default=str(DATA), help="Output root directory (default: ./data)")
    args = ap.parse_args()

    outroot = Path(args.out)
    if args.all or args.sn:
        download_pantheon_plus(outroot / "pantheon_plus")
    if args.all or args.sdss:
        download_sdss_bao(outroot / "sdss_bao")
    if args.all or args.desi:
        download_desi_dr2(outroot / "desi_dr2")

    if not any([args.all, args.sn, args.sdss, args.desi]):
        ap.print_help()

if __name__ == "__main__":
    main()
