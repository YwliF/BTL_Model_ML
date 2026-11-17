import sys, os
sys.path.insert(0, os.path.abspath('.'))
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.data_loader import load_raw_data
from src.feature_builder import FastFeatureBuilder

def rmsle(y_true, y_pred):
    """Kaggle evaluation metric: Root Mean Squared Logarithmic Error."""
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))

print("Loading data...")
train, test, stores, oil, holidays, sample_sub = load_raw_data()
fb = FastFeatureBuilder(train, stores, oil, holidays)

cutoff_date = '2017-07-30'
eval_dates = pd.date_range('2017-07-31', '2017-08-15', freq='D')
print(f"Validation Window 1: Cutoff {cutoff_date}, Eval {eval_dates[0].date()} to {eval_dates[-1].date()} ({len(eval_dates)} days)")

# Pre-extract target history for encodings
train_history = train[train['date'] <= cutoff_date].copy()
train_history['dayofweek'] = train_history['date'].dt.dayofweek

# Features list
drop_cols = ['date', 'target', 'sales']

# Test Step 1 Direct Model
print("\n--- Testing Direct Model for Step 1 ---")
t0 = time.time()
tr_feat, ev_feat = fb.build_direct_features_for_cutoff(cutoff_date, horizon_step=1, train_start_date='2016-01-01', target_df=train)
tr_enriched = fb.enrich_features(tr_feat, cutoff_date, train_history)
ev_enriched = fb.enrich_features(ev_feat, cutoff_date, train_history)

feature_cols = [c for c in tr_enriched.columns if c not in drop_cols]
cat_cols = ['store_nbr', 'family', 'city', 'state', 'type', 'cluster', 'national_h_type']
for c in cat_cols:
    if c in tr_enriched.columns:
        tr_enriched[c] = tr_enriched[c].astype('category')
        ev_enriched[c] = ev_enriched[c].astype('category')

X_tr = tr_enriched[feature_cols]
y_tr = tr_enriched['target']

X_ev = ev_enriched[feature_cols]
y_ev_true = np.expm1(ev_enriched['target'].values)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'n_estimators': 350,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_jobs': -1,
    'random_state': 42,
    'verbose': -1
}

model = lgb.LGBMRegressor(**params)
model.fit(X_tr, y_tr)

pred_log = model.predict(X_ev)
pred_sales = np.expm1(pred_log)
pred_sales = np.clip(pred_sales, 0, None)

# Zero out zero pairs
zero_mask = (ev_enriched['is_zero_pair'] == 1) & (ev_enriched['onpromotion'] == 0)
pred_sales[zero_mask] = 0.0

step1_rmsle = rmsle(y_ev_true, pred_sales)
print(f"Step 1 (Day 1) Trained in {time.time()-t0:.2f}s | RMSLE: {step1_rmsle:.4f}")

