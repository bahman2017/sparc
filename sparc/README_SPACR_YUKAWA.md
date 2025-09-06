# SPARC Yukawa/Delay Fits

This repository provides scripts to fit **SPARC** rotation curves with a simple Yukawa/delay modification.

## Scripts
- `fit_sparc_yukawa.py`: minimal two-parameter fit (α_eff, λ).
- `fit_sparc_yukawa_plus.py`: enhanced fitter with optional M/L scaling (Υ_d, Υ_b), intrinsic error floor, robust loss.
- `postprocess_results.py`: summarize a `results.csv` into plots + a compact LaTeX table.

## Usage
```bash
# Minimal
python fit_sparc_yukawa.py --zip Rotmod_LTG.zip --out ./results --make-plots

# Enhanced (recommended)
python fit_sparc_yukawa_plus.py --zip Rotmod_LTG.zip --out ./results_plus \
  --fit-ml --ml-prior 1.0,0.3 --ml-bounds 0.3,2.0 \
  --fit-errfloor --robust \
  --lambda-bounds 1,50 --alpha-bounds 0,5 \
  --make-plots

# Postprocess
python postprocess_results.py --results ./results_plus/results.csv --out ./results_plus
```

## Model
\(
v_\mathrm{model}^2(r)=\big[\Upsilon_d\,v_\mathrm{disk}^2 + \Upsilon_b\,v_\mathrm{bulge}^2 + v_\mathrm{gas}^2\big]\,
\Big[1+\alpha_\mathrm{eff}\,(1+r/\lambda)\,e^{-r/\lambda}\Big]
\)

## Requirements
```
numpy
pandas
matplotlib
scipy   # optional but faster
```
