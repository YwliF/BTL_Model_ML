import sys, os
sys.path.insert(0, os.path.abspath('.'))
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.data_loader import load_raw_data
from src.feature_builder import FastFeatureBuilder

def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))

print("Loading data...")
train, test, stores, oil, holidays, sample_sub = load_raw_data()
fb = FastFeatureBuilder(train, stores, oil, holidays)

cutoff_date = '2017-07-30'
eval_dates = pd.date_range('2017-07-31', '2017-08-15', freq='D')
n_days = len(eval_dates)

train_history = train[train['date'] <= cutoff_date].copy()
train_history['dayofweek'] = train_history['date'].dt.dayofweek

drop_cols = ['date', 'target', 'sales']
cat_cols = ['store_nbr', 'family', 'city', 'state', 'type', 'cluster', 'national_h_type']

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.06,
    'n_estimators': 300,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_jobs': -1,
    'random_state': 42,
    'verbose': -1
}

# ----------------- STRATEGY 1: HORIZON-AGNOSTIC (LAG >= 16) -----------------
print("\n=======================================================")
print("EVALUATING STRATEGY 1: HORIZON-AGNOSTIC MODEL (LAG >= 16)")
print("=======================================================")
t0 = time.time()
tr_feat, ev_feat = fb.build_lag16_features_for_cutoff(cutoff_date, train_start_date='2016-08-01', eval_dates=eval_dates)
tr_enriched = fb.enrich_features(tr_feat, cutoff_date, train_history)
ev_enriched = fb.enrich_features(ev_feat, cutoff_date, train_history)

for c in cat_cols:
    if c in tr_enriched.columns:
        tr_enriched[c] = tr_enriched[c].astype('category')
        ev_enriched[c] = ev_enriched[c].astype('category')

feature_cols = [c for c in tr_enriched.columns if c not in drop_cols]
X_tr = tr_enriched[feature_cols]
y_tr = tr_enriched['target']

model_agnostic = lgb.LGBMRegressor(**lgb_params)
model_agnostic.fit(X_tr, y_tr)

pred_log_agnostic = model_agnostic.predict(ev_enriched[feature_cols])
pred_sales_agnostic = np.clip(np.expm1(pred_log_agnostic), 0, None)

# Zero out zero pairs
zero_mask_ev = (ev_enriched['is_zero_pair'] == 1) & (ev_enriched['onpromotion'] == 0)
pred_sales_agnostic[zero_mask_ev] = 0.0

# Extract ground truth
ev_merged = ev_enriched[['date', 'store_nbr', 'family']].merge(train[['date', 'store_nbr', 'family', 'sales']], on=['date', 'store_nbr', 'family'], how='left')
y_true_all = ev_merged['sales'].fillna(0).values

score_agnostic = rmsle(y_true_all, pred_sales_agnostic)
time_agnostic = time.time() - t0
print(f"Strategy 1 Done in {time_agnostic:.1f}s | Full 16-Day Window RMSLE: {score_agnostic:.4f}")

# ----------------- STRATEGY 2: DIRECT MULTI-HORIZON (16 MODELS) -----------------
print("\n=======================================================")
print("EVALUATING STRATEGY 2: DIRECT MULTI-HORIZON (16 MODELS)")
print("=======================================================")
t0 = time.time()
direct_preds = []

for h in range(1, 17):
    th0 = time.time()
    tr_feat_h, ev_feat_h = fb.build_direct_features_for_cutoff(cutoff_date, horizon_step=h, train_start_date='2016-08-01')
    tr_en_h = fb.enrich_features(tr_feat_h, cutoff_date, train_history)
    ev_en_h = fb.enrich_features(ev_feat_h, cutoff_date, train_history)
    
    for c in cat_cols:
        if c in tr_en_h.columns:
            tr_en_h[c] = tr_en_h[c].astype('category')
            ev_en_h[c] = ev_en_h[c].astype('category')
            
    feat_cols_h = [c for c in tr_en_h.columns if c not in drop_cols]
    
    model_h = lgb.LGBMRegressor(**lgb_params)
    model_h.fit(tr_en_h[feat_cols_h], tr_en_h['target'])
    
    p_log = model_h.predict(ev_en_h[feat_cols_h])
    p_sales = np.clip(np.expm1(p_log), 0, None)
    
    zm = (ev_en_h['is_zero_pair'] == 1) & (ev_en_h['onpromotion'] == 0)
    p_sales[zm] = 0.0
    
    ev_en_h['pred_sales'] = p_sales
    direct_preds.append(ev_en_h[['date', 'store_nbr', 'family', 'pred_sales']])
    
    # Calculate step h rmsle
    true_h = ev_en_h.merge(train[['date', 'store_nbr', 'family', 'sales']], on=['date', 'store_nbr', 'family'], how='left')['sales'].fillna(0).values
    step_score = rmsle(true_h, p_sales)
    print(f"  Horizon Day {h:2d} ({ev_en_h['date'].iloc[0].strftime('%Y-%m-%d')}) | RMSLE: {step_score:.4f} | {time.time()-th0:.1f}s")

df_direct_pred = pd.concat(direct_preds, ignore_index=True)
# Align with true
df_direct_eval = ev_merged[['date', 'store_nbr', 'family', 'sales']].merge(df_direct_pred, on=['date', 'store_nbr', 'family'], how='left')
score_direct = rmsle(df_direct_eval['sales'].values, df_direct_eval['pred_sales'].values)
time_direct = time.time() - t0
print(f"Strategy 2 Done in {time_direct:.1f}s | Full 16-Day Window RMSLE: {score_direct:.4f}")

# ----------------- STRATEGY 3: ENSEMBLE (WEIGHTED BLEND) -----------------
print("\n=======================================================")
print("EVALUATING STRATEGY 3: ENSEMBLE BLEND (DIRECT + AGNOSTIC)")
print("=======================================================")
for w in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    blend = w * df_direct_eval['pred_sales'].values + (1.0 - w) * pred_sales_agnostic
    b_score = rmsle(y_true_all, blend)
    print(f"  Weight Direct={w:.1f}, Agnostic={1-w:.1f} -> RMSLE: {b_score:.4f}")

