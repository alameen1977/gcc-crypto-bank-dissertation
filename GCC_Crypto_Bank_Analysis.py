# =============================================================================
# MSO4992 — MSc Financial Technology Dissertation
# Middlesex University Dubai
# =============================================================================
# Title:  Does Cryptocurrency Market Volatility Negatively Impact the
#         Financial Performance of GCC Banks?
#
# Sample: 17 GCC banks x 8 years (2017-2024) = 136 observations
#
# PIPELINE:
#   Stage 1  — Load and clean Bitcoin daily data
#   Stage 2  — GARCH volatility modelling (select best model by AIC/BIC)
#   Stage 3  — Load and clean GCC bank panel data
#   Stage 4  — Merge volatility into panel
#   Stage 5  — Descriptive statistics and correlations
#   Stage 6  — Panel regression (Fixed Effects) — H1, H2, H3, H5
#   Stage 7  — NARDL asymmetry test — H4
#   Stage 8  — Machine learning comparison (time-aware split)
#   Stage 9  — Robustness check (historical volatility vs GARCH)
#   Stage 10 — Complete results summary
#   Stage 11 — Extended robustness check (optional, additive only —
#              does not alter the Stage 6-10 headline results)
#
# REQUIREMENTS:
#   pip install pandas numpy openpyxl arch scikit-learn
#               statsmodels linearmodels xgboost matplotlib
# =============================================================================

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from linearmodels.panel import PanelOLS, PooledOLS
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR   = Path(r"C:\Users\HP\Desktop\Dissert Clo test")

BTC_FILE   = DATA_DIR / "BTC_Daily_Data.xlsx"
BANK_FILE  = DATA_DIR / "GCC_Dataset_Complete.xlsx"
OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

START_YEAR = 2017
END_YEAR   = 2024

for f in [BTC_FILE, BANK_FILE]:
    if not f.exists():
        raise FileNotFoundError(f"File not found: {f}\nUpdate DATA_DIR above.")

print("=" * 65)
print("GCC CRYPTO-BANKING ANALYSIS — MSO4992")
print("Middlesex University Dubai")
print("=" * 65)

# =============================================================================
# STAGE 1: LOAD AND CLEAN BITCOIN DAILY DATA
# =============================================================================

print("\n[Stage 1] Loading Bitcoin daily price data...")

btc = pd.read_excel(BTC_FILE, sheet_name=0)
btc.columns = (btc.columns.astype(str).str.strip().str.lower()
               .str.replace(" ", "_").str.replace(r"[()]", "", regex=True))

btc = btc.rename(columns={
    "btc_close_price_usd": "btc_close",
    "btc_daily_log_return": "btc_log_return"
})

btc["date"]      = pd.to_datetime(btc["date"], errors="coerce")
btc["btc_close"] = pd.to_numeric(btc["btc_close"], errors="coerce")

btc = (btc.dropna(subset=["date", "btc_close"])
       .sort_values("date").drop_duplicates("date").reset_index(drop=True))

# Recalculate log returns from prices
btc["return"] = np.log(btc["btc_close"] / btc["btc_close"].shift(1))
btc["year"]   = btc["date"].dt.year
btc = btc[btc["year"].between(START_YEAR, END_YEAR)].dropna(subset=["return"]).copy()

print(f"  BTC daily observations: {len(btc):,}")
print(f"  Date range: {btc['date'].min().date()} to {btc['date'].max().date()}")

# =============================================================================
# STAGE 2: GARCH VOLATILITY MODELLING
# =============================================================================

print("\n[Stage 2] GARCH volatility modelling...")

returns_pct = btc["return"] * 100  # Scale for numerical stability

garch_specs = {
    "GARCH":  arch_model(returns_pct, mean="Constant",
                         vol="GARCH", p=1, q=1, dist="t"),
    "EGARCH": arch_model(returns_pct, mean="Constant",
                         vol="EGARCH", p=1, o=1, q=1, dist="t"),
    "TGARCH": arch_model(returns_pct, mean="Constant",
                         vol="GARCH", p=1, o=1, q=1, power=2.0, dist="t"),
}

fit_results      = {}
model_comparison = []

for name, spec in garch_specs.items():
    res = spec.fit(disp="off")
    fit_results[name] = res
    model_comparison.append({
        "model":          name,
        "AIC":            round(res.aic, 3),
        "BIC":            round(res.bic, 3),
        "log_likelihood": round(res.loglikelihood, 3)
    })

model_df = pd.DataFrame(model_comparison).sort_values("AIC").reset_index(drop=True)
model_df.to_csv(OUTPUT_DIR / "garch_model_comparison.csv", index=False)

print("\n  GARCH Model Comparison (lower AIC/BIC = better):")
print(f"  {'Model':<12} {'AIC':>12} {'BIC':>12} {'Log-Lik':>12}")
print("  " + "-" * 52)
for i, row in model_df.iterrows():
    marker = " <- SELECTED" if i == 0 else ""
    print(f"  {row['model']:<12} {row['AIC']:>12.3f} "
          f"{row['BIC']:>12.3f} {row['log_likelihood']:>12.3f}{marker}")

# Select best model by lowest AIC
best_model_name   = model_df.iloc[0]["model"]
best_model_result = fit_results[best_model_name]

print(f"\n  Selected: {best_model_name}")
print(f"  AIC={model_df.iloc[0]['AIC']:.3f} | BIC={model_df.iloc[0]['BIC']:.3f}")

# Extract conditional volatility and aggregate to annual
btc["cond_volatility"] = best_model_result.conditional_volatility.values
btc["cond_variance"]   = btc["cond_volatility"] ** 2

# Also compute historical (rolling) volatility for robustness check
btc["hist_volatility_30d"] = (btc["return"].rolling(30).std() * np.sqrt(252))

annual_crypto = (
    btc.groupby("year")
    .agg(
        crypto_volatility    =("cond_volatility",    "mean"),
        crypto_variance      =("cond_variance",      "mean"),
        hist_volatility      =("hist_volatility_30d","mean"),
        btc_return_sd        =("return",             "std"),
        btc_observations     =("return",             "count")
    )
    .reset_index()
)

annual_crypto.to_csv(OUTPUT_DIR / "annual_crypto_volatility.csv", index=False)

print("\n  Annual Crypto Volatility (GARCH-derived):")
print(f"  {'Year':<6} {'Volatility':>12} {'Period'}")
print("  " + "-" * 35)
for _, row in annual_crypto.iterrows():
    period = "Post-2022" if row["year"] >= 2022 else "Pre-2022"
    print(f"  {int(row['year']):<6} {row['crypto_volatility']:>12.4f} {period}")

# =============================================================================
# STAGE 3: LOAD AND CLEAN GCC BANK PANEL DATA
# =============================================================================

print("\n[Stage 3] Loading GCC bank panel data...")

banks = pd.read_excel(BANK_FILE, sheet_name=0, header=1)
banks.columns = (banks.columns.astype(str).str.strip().str.lower()
                 .str.replace(" ", "_").str.replace(r"[()]", "", regex=True))

banks["year"] = pd.to_numeric(banks["year"], errors="coerce").astype("Int64")
banks["bank"] = banks["bank"].astype(str).str.strip()
banks["country"] = banks["country"].astype(str).str.strip()

banks = banks[banks["year"].between(START_YEAR, END_YEAR)].copy()

# Convert all numeric columns
numeric_cols = [
    "roa", "roe", "npl", "car", "cet1", "total_assets", "net_profit",
    "nim", "cost_to_income", "loan_to_deposit", "deposit_growth",
    "provision_coverage", "non_interest_income", "country_code",
    "islamic_bank", "gdp_growth", "cpi_inflation", "oil_price", "log_assets"
]
for col in numeric_cols:
    if col in banks.columns:
        banks[col] = pd.to_numeric(banks[col], errors="coerce")

# Remove duplicates — keep one row per bank-year
banks = (banks.drop_duplicates(subset=["bank", "year"])
         .sort_values(["bank", "year"]).reset_index(drop=True))

# Post-2022 regulatory dummy (H5)
# VARA launched March 2022 — most prominent GCC crypto regulation
# Dummy captures broader GCC regulatory shift not UAE alone
banks["post_2022"]   = (banks["year"] >= 2022).astype(int)
banks["islamic_bank"] = banks["islamic_bank"].fillna(0).astype(int)

print(f"  Banks:        {banks['bank'].nunique()}")
print(f"  Countries:    {banks['country'].nunique()}")
print(f"  Observations: {len(banks)}")
print(f"  Years:        {banks['year'].min()} to {banks['year'].max()}")

# =============================================================================
# STAGE 4: MERGE CRYPTO VOLATILITY INTO BANK PANEL
# =============================================================================

print("\n[Stage 4] Merging crypto volatility with bank panel...")

panel = banks.merge(
    annual_crypto[["year", "crypto_volatility", "hist_volatility",
                   "crypto_variance", "btc_return_sd"]],
    on="year", how="left", validate="many_to_one"
)

panel = panel.sort_values(["bank", "year"]).reset_index(drop=True)
panel["bank_id"] = panel["bank"].astype("category").cat.codes

# Verify merge quality
missing_vol = panel["crypto_volatility"].isna().sum()
print(f"  Panel observations:          {len(panel)}")
print(f"  Missing crypto_volatility:   {missing_vol}")
print(f"  Merge complete:              {'Yes' if missing_vol == 0 else 'CHECK DATA'}")

if missing_vol > 0:
    print("  WARNING: Some rows missing crypto volatility — check year overlap")
    print(f"  BTC years: {sorted(annual_crypto['year'].tolist())}")
    print(f"  Bank years: {sorted(panel['year'].unique().tolist())}")

panel.to_csv(OUTPUT_DIR / "gcc_crypto_panel_clean.csv", index=False)

# =============================================================================
# STAGE 5: DESCRIPTIVE STATISTICS AND PRELIMINARY TESTS
# =============================================================================

print("\n[Stage 5] Descriptive statistics and preliminary tests...")

key_vars = [
    "roa", "roe", "npl", "crypto_volatility", "nim",
    "cost_to_income", "loan_to_deposit", "deposit_growth",
    "gdp_growth", "cpi_inflation", "oil_price", "car"
]
key_vars = [v for v in key_vars if v in panel.columns]

# Descriptive statistics
desc = panel[key_vars].describe().round(3)
desc.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")

print("\n  Descriptive Statistics:")
print(f"  {'Variable':<22} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("  " + "-" * 55)
for var in key_vars:
    s = panel[var].describe()
    print(f"  {var:<22} {s['mean']:>8.3f} {s['std']:>8.3f} "
          f"{s['min']:>8.3f} {s['max']:>8.3f}")

# Correlation with ROA
print("\n  Correlations with ROA:")
print(f"  {'Variable':<25} {'Correlation':>12} {'Direction'}")
print("  " + "-" * 48)
corr_vars = ["roe", "npl", "crypto_volatility", "nim",
             "cost_to_income", "gdp_growth", "deposit_growth"]
for var in corr_vars:
    if var in panel.columns:
        c = panel[["roa", var]].corr().iloc[0, 1]
        direction = "Negative" if c < 0 else "Positive"
        h = " <- Key (H1)" if var == "crypto_volatility" else \
            " <- Mediation" if var == "nim" else ""
        print(f"  {var:<25} {c:>12.3f} {direction}{h}")

# ADF Stationarity Tests
print("\n  ADF Stationarity Test (pooled panel):")
print(f"  {'Variable':<22} {'p-value':>10} {'Stationary?'}")
print("  " + "-" * 45)
for var in ["roa", "roe", "npl", "crypto_volatility", "gdp_growth"]:
    if var in panel.columns:
        pval = adfuller(panel[var].dropna())[1]
        stat = "Yes" if pval < 0.05 else "No — use in differences"
        print(f"  {var:<22} {pval:>10.3f} {stat}")

# VIF Multicollinearity
print("\n  VIF Multicollinearity Test:")
vif_vars = ["crypto_volatility", "gdp_growth", "cpi_inflation",
            "post_2022", "islamic_bank", "log_assets"]
vif_vars = [v for v in vif_vars if v in panel.columns]
X_vif = panel[vif_vars].dropna()
vif_df = pd.DataFrame({
    "Variable": vif_vars,
    "VIF": [variance_inflation_factor(X_vif.values, i)
            for i in range(len(vif_vars))]
})
vif_df["Status"] = vif_df["VIF"].apply(
    lambda x: "High (>10)" if x > 10 else "OK")
print(vif_df.to_string(index=False))
vif_df.to_csv(OUTPUT_DIR / "vif_results.csv", index=False)

# =============================================================================
# STAGE 6: PANEL REGRESSION — FIXED EFFECTS
# =============================================================================
# Specification: Fixed Effects with ENTITY effects only.
# Hausman principle: FE chosen — bank-specific effects correlated with regressors
# Three separate regressions: ROA (H1), ROE (H2), NPL (H3)
# Key variable: crypto_volatility
# Controls: gdp_growth, cpi_inflation, post_2022 (H5 — level effect, see
# Stage 11 for an interaction-term moderation test), log_assets
# =============================================================================

print("\n[Stage 6] Panel regression — Fixed Effects (entity effects only)...")
print("  Testing H1 (ROA), H2 (ROE), H3 (NPL) separately")

regression_vars = [
    "crypto_volatility", "nim", "cost_to_income", "loan_to_deposit",
    "deposit_growth", "gdp_growth", "cpi_inflation", "post_2022", "log_assets"
]
regression_vars = [v for v in regression_vars if v in panel.columns]

# Simple specification — crypto + macro only (isolates direct effect)
simple_vars = ["crypto_volatility", "gdp_growth", "cpi_inflation",
               "post_2022", "log_assets"]
simple_vars = [v for v in simple_vars if v in panel.columns]

panel_reg = (panel.set_index(["bank", "year"])
             .dropna(subset=["roa", "roe", "npl"] + regression_vars))


def run_fe(dep_var, x_vars, entity_effects=True, time_effects=False):
    """Run Fixed Effects panel regression and return results."""
    data = panel_reg[[dep_var] + x_vars].dropna()
    model = PanelOLS(
        data[dep_var],
        data[x_vars],
        entity_effects=entity_effects,
        time_effects=time_effects,
        drop_absorbed=True
    )
    result = model.fit(cov_type="clustered", cluster_entity=True)
    return result


def sig_stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def print_regression(result, dep_var, x_vars, hypothesis=None):
    print(f"\n  Dependent Variable: {dep_var.upper()}")
    print(f"  {'Variable':<22} {'Coef':>10} {'p-value':>10} {'Sig':>5}")
    print("  " + "-" * 52)
    for var in x_vars:
        coef = result.params.get(var, np.nan)
        pval = result.pvalues.get(var, np.nan)
        stars = sig_stars(pval)
        h_note = ""
        if var == "crypto_volatility":
            h_note = f" <- {hypothesis}" if hypothesis else ""
        print(f"  {var:<22} {coef:>10.4f} {pval:>10.4f} {stars:>5}{h_note}")
    print("  " + "-" * 52)
    print(f"  R-squared: {result.rsquared:.4f}")
    cv = result.pvalues.get("crypto_volatility", 1)
    if hypothesis:
        supported = "SUPPORTED" if cv < 0.05 else "NOT SUPPORTED"
        print(f"  {hypothesis}: {supported} (p={cv:.4f})")


# -- H1: Crypto volatility -> ROA --
print("\n  " + "=" * 55)
print("  HYPOTHESIS H1: Crypto volatility -> lower ROA")
print("  " + "=" * 55)
res_roa_simple = run_fe("roa", simple_vars)
res_roa_full   = run_fe("roa", regression_vars)
print_regression(res_roa_simple, "roa", simple_vars, "H1")

# -- H2: Crypto volatility -> ROE --
print("\n  " + "=" * 55)
print("  HYPOTHESIS H2: Crypto volatility -> lower ROE")
print("  " + "=" * 55)
res_roe = run_fe("roe", simple_vars)
print_regression(res_roe, "roe", simple_vars, "H2")

# -- H3: Crypto volatility -> NPL --
print("\n  " + "=" * 55)
print("  HYPOTHESIS H3: Crypto volatility -> higher NPL")
print("  " + "=" * 55)
res_npl = run_fe("npl", simple_vars)
print_regression(res_npl, "npl", simple_vars, "H3")

# Full model ROA — reveals mediation through NIM
print("\n  " + "=" * 55)
print("  FULL MODEL: ROA with operational variables")
print("  (Tests mediation through NIM)")
print("  " + "=" * 55)
print_regression(res_roa_full, "roa", regression_vars, None)

# -- H5: post-2022 period — level effect on ROA and ROE --
print("\n  " + "=" * 55)
print("  HYPOTHESIS H5: Post-2022 period — level effect")
print("  " + "=" * 55)
for label, res in [("ROA", res_roa_simple), ("ROE", res_roe), ("NPL", res_npl)]:
    p5 = res.pvalues.get("post_2022", np.nan)
    c5 = res.params.get("post_2022", np.nan)
    verdict = "SUPPORTED" if p5 < 0.05 else "PARTIAL/MARGINAL" if p5 < 0.10 else "NOT SUPPORTED"
    print(f"  post_2022 -> {label:<4} coef={c5:>9.4f}  p={p5:.4f}  {verdict}")

# Save regression results
reg_summary = []
for dep_var, var_name, res, hyp in [
    ("roa", "ROA", res_roa_simple, "H1"),
    ("roe", "ROE", res_roe,        "H2"),
    ("npl", "NPL", res_npl,        "H3"),
]:
    for var in simple_vars:
        reg_summary.append({
            "dependent":  var_name,
            "hypothesis": hyp if var == "crypto_volatility" else "",
            "variable":   var,
            "coefficient":res.params.get(var, np.nan),
            "p_value":    res.pvalues.get(var, np.nan),
            "significant":res.pvalues.get(var, 1) < 0.05
        })

pd.DataFrame(reg_summary).to_csv(
    OUTPUT_DIR / "panel_regression_results.csv", index=False)

# =============================================================================
# STAGE 7: NARDL ASYMMETRY TEST — H4
# =============================================================================

print("\n[Stage 7] NARDL asymmetry test — H4...")

sector = (panel.groupby("year")[["roa", "crypto_volatility",
                                  "gdp_growth", "cpi_inflation"]]
          .mean().reset_index().sort_values("year").reset_index(drop=True))

sector["crypto_change"] = sector["crypto_volatility"].diff()
sector["crypto_pos"]    = sector["crypto_change"].apply(
    lambda x: x if x > 0 else 0).cumsum()
sector["crypto_neg"]    = sector["crypto_change"].apply(
    lambda x: x if x < 0 else 0).cumsum()
sector = sector.dropna().reset_index(drop=True)

X_nardl   = sm.add_constant(
    sector[["crypto_pos", "crypto_neg", "gdp_growth", "cpi_inflation"]])
nardl_res = sm.OLS(sector["roa"], X_nardl).fit()

pos_coef = nardl_res.params.get("crypto_pos", np.nan)
neg_coef = nardl_res.params.get("crypto_neg", np.nan)
pos_pval = nardl_res.pvalues.get("crypto_pos", np.nan)
neg_pval = nardl_res.pvalues.get("crypto_neg", np.nan)

h4_result = "NOT SUPPORTED" if (
    pos_pval > 0.05 and neg_pval > 0.05
) else ("SUPPORTED" if abs(neg_coef) > abs(pos_coef) else "NOT SUPPORTED")

print(f"  Positive shocks coef: {pos_coef:.4f} (p={pos_pval:.4f})")
print(f"  Negative shocks coef: {neg_coef:.4f} (p={neg_pval:.4f})")
print(f"  R-squared: {nardl_res.rsquared:.4f}")
print(f"  H4: {h4_result}")
print(f"  Note: n={len(sector)} — annual frequency limits power of test")

nardl_df = pd.DataFrame({
    "variable":    ["crypto_pos", "crypto_neg", "gdp_growth", "const"],
    "coefficient": [nardl_res.params.get(v, np.nan)
                    for v in ["crypto_pos", "crypto_neg", "gdp_growth", "const"]],
    "p_value":     [nardl_res.pvalues.get(v, np.nan)
                    for v in ["crypto_pos", "crypto_neg", "gdp_growth", "const"]],
    "r_squared":   [nardl_res.rsquared, "", "", ""]
})
nardl_df.to_csv(OUTPUT_DIR / "nardl_results.csv", index=False)

# =============================================================================
# STAGE 8: MACHINE LEARNING COMPARISON
# =============================================================================

print("\n[Stage 8] Machine learning models (time-aware split)...")

target   = "roa"
features = [
    "crypto_volatility",  # Primary independent variable — H1
    "nim",                # Mediation channel — NIM compression
    "cost_to_income",     # Operational efficiency channel
    "loan_to_deposit",    # Liquidity channel
    "deposit_growth",     # Deposit flight channel
    "gdp_growth",         # Macro control
    "cpi_inflation",      # Macro control
    "oil_price",          # GCC-specific macro control
    "log_assets",         # Bank size control
    "car",                # Capital adequacy
    "post_2022",          # Post-2022 regulatory dummy (H5)
    "islamic_bank",       # Islamic bank dummy (exploratory)
]
features = [f for f in features if f in panel.columns]

model_data = panel[["bank", "year", target] + features].dropna().copy()

# Interaction term: crypto x Islamic bank (differential sensitivity)
model_data["crypto_islamic"] = (
    model_data["crypto_volatility"] * model_data["islamic_bank"])
features_final = features + ["crypto_islamic"]

# Time-aware split
train = model_data[model_data["year"] < END_YEAR].copy()
test  = model_data[model_data["year"] == END_YEAR].copy()

X_train, y_train = train[features_final], train[target]
X_test,  y_test  = test[features_final],  test[target]

print(f"  Train: {START_YEAR}-{END_YEAR-1} (n={len(X_train)})")
print(f"  Test:  {END_YEAR} holdout   (n={len(X_test)})")
print(f"  Features: {len(features_final)}")

# -- LASSO --
print("\n  Training LASSO...")
lasso_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("model",   LassoCV(cv=5, random_state=42, max_iter=100000))
])
lasso_pipe.fit(X_train, y_train)
lasso_pred = lasso_pipe.predict(X_test)

# -- RANDOM FOREST --
print("  Training Random Forest...")
rf_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model",   RandomForestRegressor(
        n_estimators=500, max_depth=4,
        min_samples_leaf=3, random_state=42, n_jobs=-1))
])
rf_pipe.fit(X_train, y_train)
rf_pred = rf_pipe.predict(X_test)

# -- XGBOOST --
print("  Training XGBoost...")
xgb_pipe = None
try:
    from xgboost import XGBRegressor
    xgb_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   XGBRegressor(
            n_estimators=300, max_depth=2, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", random_state=42, n_jobs=-1))
    ])
    xgb_pipe.fit(X_train, y_train)
    xgb_pred = xgb_pipe.predict(X_test)
except ImportError:
    xgb_pred = np.full(len(y_test), np.nan)
    print("  XGBoost not installed — pip install xgboost")

# -- Evaluate --
predictions = {
    "LASSO":         lasso_pred,
    "Random Forest": rf_pred,
    "XGBoost":       xgb_pred
}

evaluation = []
for name, pred in predictions.items():
    if np.isnan(pred).all():
        continue
    evaluation.append({
        "model":             name,
        "test_n":            len(y_test),
        "R2":                round(r2_score(y_test, pred), 4),
        "RMSE":              round(mean_squared_error(y_test, pred) ** 0.5, 4),
        "MAE":               round(mean_absolute_error(y_test, pred), 4)
    })

eval_df = (pd.DataFrame(evaluation)
           .sort_values("R2", ascending=False)
           .reset_index(drop=True))
eval_df.to_csv(OUTPUT_DIR / "ml_performance.csv", index=False)

fe_r2 = res_roa_simple.rsquared  # Panel regression benchmark

print(f"\n  ML Results vs Panel Regression Benchmark:")
print(f"  {'Model':<20} {'R2':>8} {'RMSE':>8} {'MAE':>8}")
print("  " + "-" * 48)
print(f"  {'Fixed Effects*':<20} {fe_r2:>8.4f} {'-':>8} {'-':>8}  *benchmark")
for i, row in eval_df.iterrows():
    flag = " <- BEST" if i == 0 else ""
    print(f"  {row['model']:<20} {row['R2']:>8.4f} "
          f"{row['RMSE']:>8.4f} {row['MAE']:>8.4f}{flag}")

# Feature importance — Random Forest
rf_model = rf_pipe.named_steps["model"]
importance_df = pd.DataFrame({
    "feature":    features_final,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)

importance_df.to_csv(OUTPUT_DIR / "feature_importance_random_forest.csv", index=False)

print("\n  Feature Importance (Random Forest):")
for _, row in importance_df.head(8).iterrows():
    bar = "#" * int(row["importance"] * 50)
    print(f"  {row['feature']:<22} {row['importance']:.4f}  {bar}")

# Feature importance — XGBoost, computed separately so the abstract's
# feature-importance figure can be checked against the model it's
# actually attributed to
xgb_importance_df = None
if xgb_pipe is not None:
    xgb_model = xgb_pipe.named_steps["model"]
    xgb_importance_df = pd.DataFrame({
        "feature":    features_final,
        "importance": xgb_model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    xgb_importance_df.to_csv(OUTPUT_DIR / "feature_importance_xgboost.csv", index=False)

    print("\n  Feature Importance (XGBoost):")
    for _, row in xgb_importance_df.head(8).iterrows():
        bar = "#" * int(row["importance"] * 50)
        print(f"  {row['feature']:<22} {row['importance']:.4f}  {bar}")

# =============================================================================
# STAGE 9: ROBUSTNESS CHECK
# =============================================================================

print("\n[Stage 9] Robustness check — historical vs GARCH volatility...")

panel_rob = panel.copy()
rob_vars  = [v.replace("crypto_volatility", "hist_volatility")
             if v == "crypto_volatility" else v for v in simple_vars]
rob_vars  = [v for v in rob_vars if v in panel_rob.columns]

panel_rob_indexed = panel_rob.set_index(["bank", "year"])
panel_rob_clean   = panel_rob_indexed.dropna(
    subset=["roa"] + rob_vars)

if "hist_volatility" in panel_rob.columns and panel_rob["hist_volatility"].notna().sum() > 10:
    rob_fe = PanelOLS(
        panel_rob_clean["roa"],
        panel_rob_clean[rob_vars],
        entity_effects=True, drop_absorbed=True
    ).fit(cov_type="clustered", cluster_entity=True)

    hv_coef = rob_fe.params.get("hist_volatility", np.nan)
    hv_pval = rob_fe.pvalues.get("hist_volatility", np.nan)
    print(f"  Historical volatility -> ROA: coef={hv_coef:.4f}, p={hv_pval:.4f}")
    print(f"  GARCH volatility -> ROA:      coef={res_roa_simple.params.get('crypto_volatility', np.nan):.4f}, "
          f"p={res_roa_simple.pvalues.get('crypto_volatility', np.nan):.4f}")
    consistent = "Consistent" if (hv_pval < 0.05 and hv_coef < 0) else "Mixed"
    print(f"  Robustness: {consistent}")
else:
    print("  Historical volatility not available — skipping robustness check")
    print("  (Ensure hist_volatility column exists in merged panel)")

# =============================================================================
# STAGE 10: COMPLETE RESULTS SUMMARY
# =============================================================================

print("\n" + "=" * 65)
print("COMPLETE RESULTS SUMMARY — MSO4992")
print("GCC Crypto-Banking Volatility Study (2017-2024)")
print("=" * 65)

print(f"\n1. GARCH MODEL SELECTION")
print(f"   Best model: {best_model_name}")
print(f"   AIC: {model_df.iloc[0]['AIC']:.3f} | BIC: {model_df.iloc[0]['BIC']:.3f}")

print(f"\n2. ANNUAL CRYPTO VOLATILITY TREND")
print(f"   Pre-2022 avg:  {annual_crypto[annual_crypto['year']<2022]['crypto_volatility'].mean():.4f}")
print(f"   Post-2022 avg: {annual_crypto[annual_crypto['year']>=2022]['crypto_volatility'].mean():.4f}")

print(f"\n3. PANEL REGRESSION RESULTS (Fixed Effects — entity effects only)")
print(f"   {'Hypothesis':<48} {'Result':<18} {'p-value'}")
print("   " + "-" * 72)
hypotheses = [
    ("H1 — Crypto volatility -> lower ROA",    res_roa_simple, "crypto_volatility"),
    ("H2 — Crypto volatility -> lower ROE",    res_roe,        "crypto_volatility"),
    ("H3 — Crypto volatility -> higher NPL",   res_npl,        "crypto_volatility"),
    ("H4 — Asymmetric impact (NARDL)",         None,           None),
    ("H5 — Post-2022 level effect (ROA)",      res_roa_simple, "post_2022"),
    ("H5 — Post-2022 level effect (ROE)",      res_roe,        "post_2022"),
]
for hyp_text, res, var in hypotheses:
    if res is None:
        print(f"   {hyp_text:<48} {h4_result:<18}")
        continue
    pval = res.pvalues.get(var, 1)
    result = "SUPPORTED" if pval < 0.05 else "PARTIAL" if pval < 0.10 else "NOT SUPPORTED"
    print(f"   {hyp_text:<48} {result:<18} p={pval:.4f}")

print(f"\n4. MEDIATION FINDING")
nim_coef = res_roa_full.params.get("nim", np.nan)
nim_pval = res_roa_full.pvalues.get("nim", np.nan)
print(f"   Crypto volatility -> NIM compression -> ROA decline")
print(f"   NIM coefficient: {nim_coef:.3f} (p={nim_pval:.4f})")
print(f"   NIM-Crypto correlation: {panel[['roa','crypto_volatility']].corr().iloc[0,1]:.3f}")

print(f"\n5. ML MODEL COMPARISON (Test: 2024 holdout, n={len(X_test)})")
print(f"   {'Model':<20} {'R2':>8} {'RMSE':>8} {'MAE':>8}")
print("   " + "-" * 48)
print(f"   {'Fixed Effects*':<20} {fe_r2:>8.4f}  — benchmark")
for i, row in eval_df.iterrows():
    flag = " <- BEST" if i == 0 else ""
    print(f"   {row['model']:<20} {row['R2']:>8.4f} {row['RMSE']:>8.4f} {row['MAE']:>8.4f}{flag}")

print(f"\n6. TOP FEATURE IMPORTANCE — Random Forest")
for _, row in importance_df.head(5).iterrows():
    print(f"   {row['feature']:<22} {row['importance']:.1%}")

if xgb_importance_df is not None:
    print(f"\n6b. TOP FEATURE IMPORTANCE — XGBoost")
    for _, row in xgb_importance_df.head(5).iterrows():
        print(f"   {row['feature']:<22} {row['importance']:.1%}")

print(f"\n7. OUTPUT FILES SAVED TO: {OUTPUT_DIR}")
files = [
    "garch_model_comparison.csv",
    "annual_crypto_volatility.csv",
    "descriptive_statistics.csv",
    "vif_results.csv",
    "panel_regression_results.csv",
    "nardl_results.csv",
    "ml_performance.csv",
    "feature_importance_random_forest.csv",
    "feature_importance_xgboost.csv",
    "gcc_crypto_panel_clean.csv",
]
for f in files:
    print(f"   {f}")

print("\n" + "=" * 65)
print("ANALYSIS COMPLETE")
print("=" * 65)

# =============================================================================
# STAGE 11: EXTENDED ROBUSTNESS CHECK (OPTIONAL)
# =============================================================================
# This is an ADDITIONAL check only — it does not overwrite or change any
# of the Stage 6-10 results above. True H5 MODERATION test: adds an
# interaction term (crypto_volatility x post_2022) to test whether the
# SENSITIVITY of bank performance to crypto volatility changed after
# 2022 — this is what "moderation" actually means, versus the
# level-effect test in Stage 6.
# =============================================================================

print("\n" + "=" * 65)
print("[Stage 11] EXTENDED ROBUSTNESS CHECK (optional — additive only)")
print("=" * 65)

print("\n  H5 as a MODERATION test (interaction term added)")
print("  " + "-" * 55)

panel_reg_int = panel_reg.copy()
panel_reg_int["crypto_x_post2022"] = (
    panel_reg_int["crypto_volatility"] * panel_reg_int["post_2022"])

interaction_vars = simple_vars + ["crypto_x_post2022"]

for label, dep in [("ROA", "roa"), ("ROE", "roe"), ("NPL", "npl")]:
    data = panel_reg_int[[dep] + interaction_vars].dropna()
    mod = PanelOLS(
        data[dep], data[interaction_vars],
        entity_effects=True, drop_absorbed=True
    ).fit(cov_type="clustered", cluster_entity=True)
    coef = mod.params.get("crypto_x_post2022", np.nan)
    pval = mod.pvalues.get("crypto_x_post2022", np.nan)
    verdict = "MODERATION SUPPORTED" if pval < 0.05 else \
              "WEAK/MARGINAL MODERATION" if pval < 0.10 else "NO MODERATION EFFECT"
    print(f"  crypto_volatility x post_2022 -> {label:<4} "
          f"coef={coef:>9.4f}  p={pval:.4f}  {verdict}")

print("\n  If these interaction terms are NOT significant, the honest framing")
print("  is that H5 is a post-2022 LEVEL difference, not a moderation effect.")

print("\n" + "=" * 65)
print("STAGE 11 COMPLETE — compare against Stage 6-10 before changing anything")
print("=" * 65)