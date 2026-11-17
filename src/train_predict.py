import sys, os
sys.path.insert(0, os.path.abspath('.'))
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.data_loader import load_raw_data
from src.feature_builder import FastFeatureBuilder

def generate_submission():
    print("=======================================================")
    print("BUILDING FINAL HIGH-QUALITY PREDICTIONS & SUBMISSION")
    print("=======================================================")
    t_start = time.time()
    
    # 1. Load data
    train, test, stores, oil, holidays, sample_sub = load_raw_data()
    fb = FastFeatureBuilder(train, stores, oil, holidays)
    fb.update_test_promo(test)

    # Cutoff date is the end of train: 2017-08-15
    cutoff_date = '2017-08-15'
    eval_dates = pd.date_range('2017-08-16', '2017-08-31', freq='D')
    print(f"Train cutoff: {cutoff_date} | Forecast horizon: {eval_dates[0].date()} to {eval_dates[-1].date()} ({len(eval_dates)} days)")

    train_history = train.copy()
    train_history['dayofweek'] = train_history['date'].dt.dayofweek

    drop_cols = ['date', 'target', 'sales']
    cat_cols = ['store_nbr', 'family', 'city', 'state', 'type', 'cluster', 'national_h_type']

    # Load tuned parameters if available
    param_path = 'data/best_hyperparams.json'
    if os.path.exists(param_path):
        with open(param_path) as f:
            tuned = json.load(f)
        print("Using Optuna tuned parameters:", tuned)
    else:
        tuned = {
            'num_leaves': 63,
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 350,
            'min_child_samples': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        }

    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    }
    lgb_params.update(tuned)

    # ---------------- 1. TRAIN & PREDICT HORIZON-AGNOSTIC MODEL ----------------
    print("\n--- Training Model 1: Horizon-Agnostic Global Model (Lag >= 16) ---")
    t0 = time.time()
    tr_feat_agn, ev_feat_agn = fb.build_lag16_features_for_cutoff(cutoff_date, train_start_date='2016-08-01', eval_dates=eval_dates)
    tr_en_agn = fb.enrich_features(tr_feat_agn, cutoff_date, train_history)
    ev_en_agn = fb.enrich_features(ev_feat_agn, cutoff_date, train_history)

    for c in cat_cols:
        if c in tr_en_agn.columns:
            tr_en_agn[c] = tr_en_agn[c].astype('category')
            ev_en_agn[c] = ev_en_agn[c].astype('category')

    feat_cols = [c for c in tr_en_agn.columns if c not in drop_cols]
    
    model_agn = lgb.LGBMRegressor(**lgb_params)
    model_agn.fit(tr_en_agn[feat_cols], tr_en_agn['target'])

    pred_log_agn = model_agn.predict(ev_en_agn[feat_cols])
    pred_sales_agn = np.clip(np.expm1(pred_log_agn), 0, None)
    
    # Zero mask for agnostic
    zm_agn = (ev_en_agn['is_zero_pair'] == 1) & (ev_en_agn['onpromotion'] == 0)
    pred_sales_agn[zm_agn] = 0.0
    ev_en_agn['pred_agn'] = pred_sales_agn
    print(f"Model 1 trained & predicted in {time.time()-t0:.1f}s")

    # ---------------- 2. TRAIN & PREDICT 16 DIRECT MULTI-HORIZON MODELS ----------------
    print("\n--- Training Model 2: 16 Direct Multi-Horizon Step Models ---")
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
        
        zm_h = (ev_en_h['is_zero_pair'] == 1) & (ev_en_h['onpromotion'] == 0)
        p_sales[zm_h] = 0.0
        
        ev_en_h['pred_direct'] = p_sales
        direct_preds.append(ev_en_h[['date', 'store_nbr', 'family', 'pred_direct']])
        print(f"  Step {h:2d}/16 ({ev_en_h['date'].iloc[0].strftime('%Y-%m-%d')}) finished in {time.time()-th0:.1f}s")

    df_direct = pd.concat(direct_preds, ignore_index=True)
    print(f"All 16 Direct Models trained & predicted in {time.time()-t0:.1f}s")

    # ---------------- 3. BLEND ENSEMBLE ----------------
    print("\n--- Blending Models with Optimal Validation Weights (0.4 Direct + 0.6 Agnostic) ---")
    merged_preds = ev_en_agn[['date', 'store_nbr', 'family', 'pred_agn', 'is_zero_pair', 'onpromotion']].merge(
        df_direct[['date', 'store_nbr', 'family', 'pred_direct']],
        on=['date', 'store_nbr', 'family'],
        how='left'
    )
    
    # Blended prediction
    merged_preds['sales'] = 0.4 * merged_preds['pred_direct'] + 0.6 * merged_preds['pred_agn']
    
    # Final safety zero check
    zero_mask_final = (merged_preds['is_zero_pair'] == 1) & (merged_preds['onpromotion'] == 0)
    merged_preds.loc[zero_mask_final, 'sales'] = 0.0
    merged_preds['sales'] = np.clip(merged_preds['sales'], 0, None)

    # ---------------- 4. MERGE WITH TEST & VERIFY ----------------
    print("\n--- Merging with test.csv and verifying submission integrity ---")
    sub = test[['id', 'date', 'store_nbr', 'family']].merge(
        merged_preds[['date', 'store_nbr', 'family', 'sales']],
        on=['date', 'store_nbr', 'family'],
        how='left'
    )
    
    sub_final = sub[['id', 'sales']].copy()
    
    # Strict validation checks
    assert len(sub_final) == len(sample_sub), f"Row count mismatch: {len(sub_final)} vs {len(sample_sub)}"
    assert sub_final['sales'].isna().sum() == 0, f"Found {sub_final['sales'].isna().sum()} NaNs!"
    assert np.isinf(sub_final['sales']).sum() == 0, "Found infinite values!"
    assert (sub_final['sales'] < 0).sum() == 0, "Found negative sales!"
    assert (sub_final['id'].values == sample_sub['id'].values).all(), "ID sequence does not match sample_submission!"
    assert list(sub_final.columns) == ['id', 'sales'], f"Columns mismatch: {list(sub_final.columns)}"

    # Save submission files
    sub_final.to_csv('submission.csv', index=False)
    sub_final.to_csv('data/submission.csv', index=False)
    print("SUCCESS: submission.csv successfully generated and verified!")
    print(f"Total rows: {len(sub_final):,}")
    print("First 10 rows:\n", sub_final.head(10).to_string())
    print("Summary stats of predicted sales:\n", sub_final['sales'].describe())
    print(f"Zero sales count in predictions: {(sub_final['sales'] == 0).sum():,} ({(sub_final['sales'] == 0).mean()*100:.2f}%)")
    print(f"\nEntire pipeline completed in {time.time()-t_start:.1f}s")

if __name__ == '__main__':
    generate_submission()

