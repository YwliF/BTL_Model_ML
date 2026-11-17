import sys, os
sys.path.insert(0, os.path.abspath('.'))
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from src.data_loader import load_raw_data
from src.feature_builder import FastFeatureBuilder

optuna.logging.set_verbosity(optuna.logging.WARNING)

def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))

print("Loading data for Optuna hyperparameter optimization...")
train, test, stores, oil, holidays, sample_sub = load_raw_data()
fb = FastFeatureBuilder(train, stores, oil, holidays)

cutoff_date = '2017-07-30'
eval_dates = pd.date_range('2017-07-31', '2017-08-15', freq='D')

train_history = train[train['date'] <= cutoff_date].copy()
train_history['dayofweek'] = train_history['date'].dt.dayofweek

drop_cols = ['date', 'target', 'sales']
cat_cols = ['store_nbr', 'family', 'city', 'state', 'type', 'cluster', 'national_h_type']

print("Building features for hyperparameter tuning...")
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
X_ev = ev_enriched[feature_cols]

ev_merged = ev_enriched[['date', 'store_nbr', 'family']].merge(
    train[['date', 'store_nbr', 'family', 'sales']], on=['date', 'store_nbr', 'family'], how='left'
)
y_true_all = ev_merged['sales'].fillna(0).values
zero_mask_ev = (ev_enriched['is_zero_pair'] == 1) & (ev_enriched['onpromotion'] == 0)

def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.09, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 250, 450, step=50),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100, step=10),
        'subsample': trial.suggest_float('subsample', 0.65, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr)
    
    pred_log = model.predict(X_ev)
    pred_sales = np.clip(np.expm1(pred_log), 0, None)
    pred_sales[zero_mask_ev] = 0.0
    
    score = rmsle(y_true_all, pred_sales)
    return score

print("Starting Optuna study (12 trials)...")
t0 = time.time()
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=12, show_progress_bar=False)

print(f"\nOptimization completed in {time.time()-t0:.1f}s")
print(f"Best trial RMSLE: {study.best_value:.4f}")
print("Best hyperparameters:")
for k, v in study.best_params.items():
    print(f"  '{k}': {v},")

# Save best params to file
import json
with open('data/best_hyperparams.json', 'w') as f:
    json.dump(study.best_params, f, indent=2)
print("Saved best params to data/best_hyperparams.json")

