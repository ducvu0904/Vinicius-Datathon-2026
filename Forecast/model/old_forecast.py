"""
Sales Forecasting - Vin Datathon 2026
Predict daily Revenue and COGS: 2023-01-01 to 2024-07-01

Key discovery: auxiliary data (orders, payments, etc.) only exists for training period.
Test period has NO auxiliary data. Must rely on:
- Calendar/time features
- Seasonal patterns (month-day profiles)
- Trend (YoY growth)
- Vietnamese calendar (Tet, holidays, sale cycles)
- Monthly cycle patterns (end-of-month spikes)
- Lag features computed from training data only

Approach:
1. Train LightGBM + XGBoost on time/calendar features using training data
2. Use recursive forecasting for test period (predict day by day, using prev predictions as lags)
3. Blend with seasonal baseline for robustness
"""

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'original-data')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. LOAD
# ============================================================
print("Loading data...")
sales = pd.read_csv(os.path.join(DATA_DIR, 'sales.csv'), parse_dates=['Date'])
sample_sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
promotions = pd.read_csv(os.path.join(DATA_DIR, 'promotions.csv'), parse_dates=['start_date', 'end_date'])

print(f"  Sales: {sales.shape}, {sales['Date'].min().date()} -> {sales['Date'].max().date()}")
print(f"  Test: {len(sample_sub)} rows, {sample_sub['Date'].iloc[0]} -> {sample_sub['Date'].iloc[-1]}")

# ============================================================
# 2. BUILD FULL DATE RANGE WITH ALL FEATURES
# ============================================================
all_dates = pd.date_range('2012-07-04', '2024-07-01', freq='D')
df = pd.DataFrame({'Date': all_dates})
df = df.merge(sales, on='Date', how='left')

# --- Calendar features ---
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['dow'] = df['Date'].dt.dayofweek  # 0=Mon, 6=Sun
df['doy'] = df['Date'].dt.dayofyear
df['woy'] = df['Date'].dt.isocalendar().week.astype(int)
df['quarter'] = df['Date'].dt.quarter
df['is_weekend'] = (df['dow'] >= 5).astype(int)
df['is_monthstart'] = df['Date'].dt.is_month_start.astype(int)
df['is_monthend'] = df['Date'].dt.is_month_end.astype(int)
df['days_in_month'] = df['Date'].dt.days_in_month
df['day_frac'] = df['day'] / df['days_in_month']

# Cyclical encoding
for c, p in [('month', 12), ('dow', 7), ('doy', 366), ('day', 31)]:
    df[f'{c}_sin'] = np.sin(2 * np.pi * df[c] / p)
    df[f'{c}_cos'] = np.cos(2 * np.pi * df[c] / p)

# Days until/from month boundaries
df['days_to_monthend'] = df['days_in_month'] - df['day']
df['days_from_monthstart'] = df['day'] - 1

# --- Vietnamese holidays and Tet ---
tet = {2012:'2012-01-23', 2013:'2013-02-10', 2014:'2014-01-31',
       2015:'2015-02-19', 2016:'2016-02-08', 2017:'2017-01-28',
       2018:'2018-02-16', 2019:'2019-02-05', 2020:'2020-01-25',
       2021:'2021-02-12', 2022:'2022-02-01', 2023:'2023-01-22',
       2024:'2024-02-10'}
tet = {k: pd.Timestamp(v) for k, v in tet.items()}

def tet_features(d):
    y = d.year
    t = tet.get(y)
    if t is None:
        return 999, 0
    diff = (t - d).days
    if diff < -15:
        tn = tet.get(y + 1)
        if tn:
            diff = (tn - d).days
    return diff, 1 if abs(diff) <= 7 else 0

df[['days_to_tet', 'is_tet_week']] = df['Date'].apply(lambda d: pd.Series(tet_features(d)))
df['abs_days_tet'] = df['days_to_tet'].abs()
df['is_tet_season'] = (df['abs_days_tet'] <= 15).astype(int)
df['is_pre_tet_30'] = ((df['days_to_tet'] >= 0) & (df['days_to_tet'] <= 30)).astype(int)
df['is_post_tet'] = ((df['days_to_tet'] >= -7) & (df['days_to_tet'] < 0)).astype(int)

# Vietnamese public holidays
def vn_holiday(d):
    m, day = d.month, d.day
    holidays = [(1,1), (4,30), (5,1), (9,2)]
    return 1 if (m, day) in holidays else 0
df['is_holiday'] = df['Date'].apply(vn_holiday)

# End-of-month sale cycle (major e-commerce pattern in Vietnam)
df['eom_sale'] = ((df['day'] >= 28) | (df['day'] <= 3)).astype(int)
df['eom_5day'] = (df['days_to_monthend'] <= 4).astype(int)
df['som_5day'] = (df['day'] <= 5).astype(int)

# Mid-month sale (11.11, etc.)
df['mid_month'] = ((df['day'] >= 10) & (df['day'] <= 12)).astype(int)

# Active promotions
for d_idx, d in enumerate(df['Date']):
    n = ((promotions['start_date'] <= d) & (promotions['end_date'] >= d)).sum()
    df.loc[d_idx, 'n_promos'] = n
df['n_promos'] = df['n_promos'].fillna(0).astype(int)

# --- Trend features ---
df['days_idx'] = (df['Date'] - df['Date'].min()).dt.days

ann_rev = sales.groupby(sales['Date'].dt.year)['Revenue'].sum()
ann_cogs = sales.groupby(sales['Date'].dt.year)['COGS'].sum()
yoy_r = ann_rev.loc[2013:2022].pct_change().dropna()
yoy_c = ann_cogs.loc[2013:2022].pct_change().dropna()
g_rev = (1 + yoy_r).prod() ** (1 / len(yoy_r))
g_cogs = (1 + yoy_c).prod() ** (1 / len(yoy_c))
print(f"  YoY growth Rev: {(g_rev-1)*100:.2f}%, COGS: {(g_cogs-1)*100:.2f}%")

df['trend_rev'] = g_rev ** (df['year'] - 2022)
df['trend_cogs'] = g_cogs ** (df['year'] - 2022)

# --- Seasonal baseline ---
tr = sales.copy()
tr['year'] = tr['Date'].dt.year; tr['month'] = tr['Date'].dt.month; tr['day'] = tr['Date'].dt.day
am = tr.groupby('year')[['Revenue','COGS']].transform('mean')
tr['rn'] = tr['Revenue'] / am['Revenue']
tr['cn'] = tr['COGS'] / am['COGS']
seasonal = tr.groupby(['month','day'])[['rn','cn']].mean().reset_index()
df = df.merge(seasonal, on=['month','day'], how='left')
df['rn'] = df['rn'].fillna(1.0); df['cn'] = df['cn'].fillna(1.0)

base_r = ann_rev.loc[2022] / 365
base_c = ann_cogs.loc[2022] / 365
df['baseline_rev'] = base_r * df['trend_rev'] * df['rn']
df['baseline_cogs'] = base_c * df['trend_cogs'] * df['cn']

# Also seasonal profiles by (month, dow) for extra pattern
seasonal_dow = tr.groupby(['month', tr['Date'].dt.dayofweek])[['rn','cn']].mean().reset_index()
seasonal_dow.columns = ['month', 'dow', 'rn_dow', 'cn_dow']
df = df.merge(seasonal_dow, on=['month', 'dow'], how='left')
df['rn_dow'] = df['rn_dow'].fillna(1.0); df['cn_dow'] = df['cn_dow'].fillna(1.0)

# Weekly seasonal profile
weekly = tr.groupby(tr['Date'].dt.dayofweek)[['rn','cn']].mean().reset_index()
weekly.columns = ['dow', 'rn_weekly', 'cn_weekly']
df = df.merge(weekly, on='dow', how='left')

# Monthly total profile
monthly = tr.groupby('month')[['rn','cn']].mean().reset_index()
monthly.columns = ['month', 'rn_monthly', 'cn_monthly']
df = df.merge(monthly, on='month', how='left')

# ============================================================
# 3. LAG FEATURES (from known Revenue/COGS only)
# ============================================================
print("Computing lag features...")
df = df.sort_values('Date').reset_index(drop=True)

# These lags only work for training. For test, we'll do recursive prediction.
for lag in [1, 2, 3, 7, 14, 28, 30, 365, 366, 730]:
    df[f'rev_lag{lag}'] = df['Revenue'].shift(lag)
    df[f'cogs_lag{lag}'] = df['COGS'].shift(lag)

for w in [7, 14, 30, 90]:
    df[f'rev_rmean{w}'] = df['Revenue'].shift(1).rolling(w, min_periods=1).mean()
    df[f'cogs_rmean{w}'] = df['COGS'].shift(1).rolling(w, min_periods=1).mean()
    df[f'rev_rstd{w}'] = df['Revenue'].shift(1).rolling(w, min_periods=1).std()

# COGS/Revenue ratio
df['cr_ratio'] = (df['COGS'] / df['Revenue'].replace(0, np.nan))
for lag in [1, 7, 30]:
    df[f'cr_lag{lag}'] = df['cr_ratio'].shift(lag)
df[f'cr_rmean7'] = df['cr_ratio'].shift(1).rolling(7, min_periods=1).mean()
df[f'cr_rmean30'] = df['cr_ratio'].shift(1).rolling(30, min_periods=1).mean()

# ============================================================
# 4. TRAIN MODEL ON HISTORICAL DATA
# ============================================================
print("Preparing training data...")
drop_cols = ['Date', 'Revenue', 'COGS', 'rn', 'cn', 'cr_ratio']
feat_cols = [c for c in df.columns if c not in drop_cols]

# Only train on data where we have Revenue + sufficient lags (from 2014+)
train_mask = df['Revenue'].notna() & (df['year'] >= 2014)
train_df = df[train_mask].copy()

# Remove features with >50% NaN in training
to_drop = [c for c in feat_cols if train_df[c].isna().sum() > len(train_df) * 0.5]
feat_cols = [c for c in feat_cols if c not in to_drop]

# Fill NaN in training
for col in feat_cols:
    med = train_df[col].median() if train_df[col].notna().any() else 0
    train_df[col] = train_df[col].fillna(med)

print(f"  Features: {len(feat_cols)}, Train: {len(train_df)}")

X_all = train_df[feat_cols]
y_all_r = train_df['Revenue']
y_all_c = train_df['COGS']

# Validation: last 549 rows (same as test size)
val_n = 549
X_tr, X_val = X_all.iloc[:-val_n], X_all.iloc[-val_n:]
y_tr_r, y_vr = y_all_r.iloc[:-val_n], y_all_r.iloc[-val_n:]
y_tr_c, y_vc = y_all_c.iloc[:-val_n], y_all_c.iloc[-val_n:]

# ============================================================
# 5. TRAIN MODELS
# ============================================================
import lightgbm as lgb
import xgboost as xgb

print("\n--- LightGBM ---")
lgb_p = dict(objective='regression', metric='mae', learning_rate=0.02,
             num_leaves=127, min_child_samples=15, subsample=0.8,
             colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=0.5,
             n_estimators=5000, random_state=SEED, verbose=-1)

m_lr = lgb.LGBMRegressor(**lgb_p)
m_lr.fit(X_tr, y_tr_r, eval_set=[(X_val, y_vr)],
         callbacks=[lgb.early_stopping(100, verbose=False)])
m_lc = lgb.LGBMRegressor(**lgb_p)
m_lc.fit(X_tr, y_tr_c, eval_set=[(X_val, y_vc)],
         callbacks=[lgb.early_stopping(100, verbose=False)])

vp_lr = m_lr.predict(X_val); vp_lc = m_lc.predict(X_val)
print(f"  Val MAE Rev: {mean_absolute_error(y_vr, vp_lr):,.0f}, R2: {r2_score(y_vr, vp_lr):.4f}")
print(f"  Val MAE COGS: {mean_absolute_error(y_vc, vp_lc):,.0f}, R2: {r2_score(y_vc, vp_lc):.4f}")
print(f"  Val RMSE Rev: {np.sqrt(mean_squared_error(y_vr, vp_lr)):,.0f}")

print("\n--- XGBoost ---")
xgb_p = dict(objective='reg:squarederror', learning_rate=0.02, max_depth=8,
             subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
             n_estimators=5000, early_stopping_rounds=100, random_state=SEED, verbosity=0)

m_xr = xgb.XGBRegressor(**xgb_p)
m_xr.fit(X_tr, y_tr_r, eval_set=[(X_val, y_vr)], verbose=False)
m_xc = xgb.XGBRegressor(**xgb_p)
m_xc.fit(X_tr, y_tr_c, eval_set=[(X_val, y_vc)], verbose=False)

vp_xr = m_xr.predict(X_val); vp_xc = m_xc.predict(X_val)
print(f"  Val MAE Rev: {mean_absolute_error(y_vr, vp_xr):,.0f}, R2: {r2_score(y_vr, vp_xr):.4f}")
print(f"  Val MAE COGS: {mean_absolute_error(y_vc, vp_xc):,.0f}, R2: {r2_score(y_vc, vp_xc):.4f}")

# Optimal ensemble weights
best_w, best_mae = 0.5, 1e18
for w in np.arange(0, 1.01, 0.05):
    mae = mean_absolute_error(y_vr, w*vp_lr + (1-w)*vp_xr)
    if mae < best_mae: best_w, best_mae = w, mae
best_wc = 0.5
best_mae_c = 1e18
for w in np.arange(0, 1.01, 0.05):
    mae = mean_absolute_error(y_vc, w*vp_lc + (1-w)*vp_xc)
    if mae < best_mae_c: best_wc, best_mae_c = w, mae
print(f"\n  Best weights - Rev: LGB={best_w:.2f}, COGS: LGB={best_wc:.2f}")

# Retrain on full training data
print("\nRetraining on full data...")
lr_full = lgb.LGBMRegressor(**{**lgb_p, 'n_estimators': m_lr.best_iteration_ or 1500})
lr_full.fit(X_all, y_all_r)
lc_full = lgb.LGBMRegressor(**{**lgb_p, 'n_estimators': m_lc.best_iteration_ or 1500})
lc_full.fit(X_all, y_all_c)

xfp = {k:v for k,v in xgb_p.items() if k != 'early_stopping_rounds'}
xr_full = xgb.XGBRegressor(**{**xfp, 'n_estimators': m_xr.best_iteration or 1500})
xr_full.fit(X_all, y_all_r, verbose=False)
xc_full = xgb.XGBRegressor(**{**xfp, 'n_estimators': m_xc.best_iteration or 1500})
xc_full.fit(X_all, y_all_c, verbose=False)

# ============================================================
# 6. RECURSIVE PREDICTION FOR TEST PERIOD
# ============================================================
print("\nRecursive forecasting for test period...")

# Get all known Revenue/COGS as a series indexed by position in df
rev_series = df['Revenue'].copy()
cogs_series = df['COGS'].copy()

test_start_idx = df[df['Revenue'].isna()].index[0]
test_end_idx = df[df['Revenue'].isna()].index[-1]

# Compute medians for filling NaN features
medians = {}
for col in feat_cols:
    medians[col] = train_df[col].median() if train_df[col].notna().any() else 0

for i in range(test_start_idx, test_end_idx + 1):
    # Recompute lag features for this row using known + predicted values
    for lag in [1, 2, 3, 7, 14, 28, 30, 365, 366, 730]:
        src_idx = i - lag
        if src_idx >= 0:
            df.at[i, f'rev_lag{lag}'] = rev_series.iloc[src_idx] if pd.notna(rev_series.iloc[src_idx]) else medians.get(f'rev_lag{lag}', 0)
            df.at[i, f'cogs_lag{lag}'] = cogs_series.iloc[src_idx] if pd.notna(cogs_series.iloc[src_idx]) else medians.get(f'cogs_lag{lag}', 0)
    
    # Rolling means from available data
    for w in [7, 14, 30, 90]:
        start = max(0, i - w)
        vals_r = rev_series.iloc[start:i]
        vals_c = cogs_series.iloc[start:i]
        valid_r = vals_r.dropna()
        valid_c = vals_c.dropna()
        df.at[i, f'rev_rmean{w}'] = valid_r.mean() if len(valid_r) > 0 else medians.get(f'rev_rmean{w}', 0)
        df.at[i, f'cogs_rmean{w}'] = valid_c.mean() if len(valid_c) > 0 else medians.get(f'cogs_rmean{w}', 0)
        df.at[i, f'rev_rstd{w}'] = valid_r.std() if len(valid_r) > 1 else medians.get(f'rev_rstd{w}', 0)
    
    # CR ratio lags
    for lag in [1, 7, 30]:
        src = i - lag
        if src >= 0 and pd.notna(rev_series.iloc[src]) and rev_series.iloc[src] != 0:
            df.at[i, f'cr_lag{lag}'] = cogs_series.iloc[src] / rev_series.iloc[src]
        else:
            df.at[i, f'cr_lag{lag}'] = medians.get(f'cr_lag{lag}', 0.8)
    
    # CR rolling means
    start7 = max(0, i - 7)
    cr_vals = []
    for j in range(start7, i):
        if pd.notna(rev_series.iloc[j]) and rev_series.iloc[j] != 0:
            cr_vals.append(cogs_series.iloc[j] / rev_series.iloc[j])
    df.at[i, 'cr_rmean7'] = np.mean(cr_vals) if cr_vals else medians.get('cr_rmean7', 0.8)
    
    start30 = max(0, i - 30)
    cr_vals30 = []
    for j in range(start30, i):
        if pd.notna(rev_series.iloc[j]) and rev_series.iloc[j] != 0:
            cr_vals30.append(cogs_series.iloc[j] / rev_series.iloc[j])
    df.at[i, 'cr_rmean30'] = np.mean(cr_vals30) if cr_vals30 else medians.get('cr_rmean30', 0.8)
    
    # Build feature row
    row = df.loc[i, feat_cols].copy()
    for col in feat_cols:
        if pd.isna(row[col]):
            row[col] = medians[col]
    
    X_row = pd.DataFrame([row], columns=feat_cols)
    
    # Predict
    p_r_lgb = lr_full.predict(X_row)[0]
    p_r_xgb = xr_full.predict(X_row)[0]
    p_c_lgb = lc_full.predict(X_row)[0]
    p_c_xgb = xc_full.predict(X_row)[0]
    
    pred_r = max(0, best_w * p_r_lgb + (1 - best_w) * p_r_xgb)
    pred_c = max(0, best_wc * p_c_lgb + (1 - best_wc) * p_c_xgb)
    
    # Store predictions for subsequent lag computation
    rev_series.iloc[i] = pred_r
    cogs_series.iloc[i] = pred_c
    
    if i % 100 == 0:
        d = df.at[i, 'Date']
        print(f"  {d.date()}: Rev={pred_r:,.0f}, COGS={pred_c:,.0f}")

# ============================================================
# 7. SUBMISSION
# ============================================================
print("\nCreating submission...")
test_df = df[df['Date'].isin(pd.to_datetime(sample_sub['Date']))].copy()
pred_rev = rev_series.loc[test_df.index].values
pred_cogs = cogs_series.loc[test_df.index].values

submission = pd.DataFrame({
    'Date': sample_sub['Date'],
    'Revenue': np.round(pred_rev, 2),
    'COGS': np.round(pred_cogs, 2),
})
assert len(submission) == len(sample_sub), f"Mismatch: {len(submission)} vs {len(sample_sub)}"

out = os.path.join(OUT_DIR, 'submission.csv')
submission.to_csv(out, index=False)
print(f"Saved {len(submission)} rows to {out}")
print(f"  Rev: {submission['Revenue'].min():,.0f} - {submission['Revenue'].max():,.0f}")
print(f"  COGS: {submission['COGS'].min():,.0f} - {submission['COGS'].max():,.0f}")
print(submission.head(10))
print("...")
print(submission.tail(5))

# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================
print("\n=== Top 25 Feature Importances (Revenue LGB) ===")
imp = pd.Series(lr_full.feature_importances_, index=feat_cols).sort_values(ascending=False).head(25)
for f, v in imp.items():
    print(f"  {f}: {v}")

print("\nDone!")
