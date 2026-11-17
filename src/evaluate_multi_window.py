import sys, os
sys.path.insert(0, os.path.abspath('.'))
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.data_loader import load_raw_data
from src.feature_builder import FastFeatureBuilder

def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))

def run_multi_window_evaluation():
    print("Loading raw data...")
    train, test, stores, oil, holidays, sample_sub = load_raw_data()
    fb = FastFeatureBuilder(train, stores, oil, holidays)

    # Load tuned parameters if available, else defaults
    param_path = 'data/best_hyperparams.json'
    if os.path.exists(param_path):
        with open(param_path) as f:
            best_tuned = json.load(f)
        print("Loaded tuned hyperparameters from:", param_path)
    else:
        best_tuned = {
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

    base_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    }
    base_params.update(best_tuned)

    drop_cols = ['date', 'target', 'sales']
    cat_cols = ['store_nbr', 'family', 'city', 'state', 'type', 'cluster', 'national_h_type']

    windows = [
        {'name': 'Window 1 (Recent)', 'cutoff': '2017-07-30', 'eval_start': '2017-07-31', 'eval_end': '2017-08-15'},
        {'name': 'Window 2 (Mid)',    'cutoff': '2017-07-14', 'eval_start': '2017-07-15', 'eval_end': '2017-07-30'},
        {'name': 'Window 3 (Early)',  'cutoff': '2017-06-28', 'eval_start': '2017-06-29', 'eval_end': '2017-07-14'}
    ]

    all_window_results = []
    all_eval_dfs = []

    for w_idx, win in enumerate(windows, 1):
        cutoff = win['cutoff']
        eval_dates = pd.date_range(win['eval_start'], win['eval_end'], freq='D')
        print(f"\n=======================================================")
        print(f"EVALUATING {win['name']}: Cutoff {cutoff} -> {win['eval_start']} to {win['eval_end']}")
        print(f"=======================================================")
        
        t0 = time.time()
        train_hist = train[train['date'] <= cutoff].copy()
        train_hist['dayofweek'] = train_hist['date'].dt.dayofweek
        
        # 1. Horizon-Agnostic Model
        tr_feat_agn, ev_feat_agn = fb.build_lag16_features_for_cutoff(cutoff, train_start_date='2016-08-01', eval_dates=eval_dates)
        tr_en_agn = fb.enrich_features(tr_feat_agn, cutoff, train_hist)
        ev_en_agn = fb.enrich_features(ev_feat_agn, cutoff, train_hist)
        
        for c in cat_cols:
            if c in tr_en_agn.columns:
                tr_en_agn[c] = tr_en_agn[c].astype('category')
                ev_en_agn[c] = ev_en_agn[c].astype('category')
                
        feat_cols = [c for c in tr_en_agn.columns if c not in drop_cols]
        
        model_agn = lgb.LGBMRegressor(**base_params)
        model_agn.fit(tr_en_agn[feat_cols], tr_en_agn['target'])
        
        pred_agn_log = model_agn.predict(ev_en_agn[feat_cols])
        pred_agn_sales = np.clip(np.expm1(pred_agn_log), 0, None)
        zm_agn = (ev_en_agn['is_zero_pair'] == 1) & (ev_en_agn['onpromotion'] == 0)
        pred_agn_sales[zm_agn] = 0.0
        
        # Ground truth
        ev_true = ev_en_agn[['date', 'store_nbr', 'family']].merge(
            train[['date', 'store_nbr', 'family', 'sales']], on=['date', 'store_nbr', 'family'], how='left'
        )
        y_true = ev_true['sales'].fillna(0).values
        
        score_agn = rmsle(y_true, pred_agn_sales)
        print(f"  {win['name']} Agnostic RMSLE: {score_agn:.4f} ({time.time()-t0:.1f}s)")
        
        # Store for detailed breakdown
        ev_eval_df = ev_true.copy()
        ev_eval_df['pred_sales'] = pred_agn_sales
        ev_eval_df['window'] = win['name']
        ev_eval_df['horizon_day'] = (ev_eval_df['date'] - pd.to_datetime(cutoff)).dt.days
        ev_eval_df['log_err_sq'] = (np.log1p(ev_eval_df['sales']) - np.log1p(ev_eval_df['pred_sales'])) ** 2
        
        all_eval_dfs.append(ev_eval_df)
        all_window_results.append({
            'window': win['name'],
            'cutoff': cutoff,
            'eval_period': f"{win['eval_start']} to {win['eval_end']}",
            'rmsle': score_agn
        })

    # Combined evaluation
    full_eval = pd.concat(all_eval_dfs, ignore_index=True)
    mean_rmsle = np.mean([r['rmsle'] for r in all_window_results])
    overall_rmsle = np.sqrt(full_eval['log_err_sq'].mean())
    
    print("\n=======================================================")
    print("STRICT MULTI-WINDOW VALIDATION SUMMARY")
    print("=======================================================")
    for r in all_window_results:
        print(f"  {r['window']} ({r['eval_period']}): RMSLE = {r['rmsle']:.4f}")
    print(f"  --> Average Window RMSLE: {mean_rmsle:.4f}")
    print(f"  --> Combined Pooled RMSLE: {overall_rmsle:.4f}")

    # Family error analysis
    print("\n=================== ERROR ANALYSIS BY FAMILY ===================")
    fam_err = full_eval.groupby('family', observed=False)['log_err_sq'].mean().apply(np.sqrt).sort_values()
    print("Top 5 lowest error (most accurate) families:")
    print(fam_err.head(5).to_string())
    print("\nTop 5 highest error (most volatile) families:")
    print(fam_err.tail(5).to_string())

    # Store error analysis
    print("\n=================== ERROR ANALYSIS BY STORE ===================")
    store_err = full_eval.groupby('store_nbr')['log_err_sq'].mean().apply(np.sqrt).sort_values()
    print("Top 5 lowest error stores:")
    print(store_err.head(5).to_string())
    print("\nTop 5 highest error stores:")
    print(store_err.tail(5).to_string())

    # Horizon day analysis
    print("\n=================== ERROR ANALYSIS BY FORECAST HORIZON ===================")
    horizon_err = full_eval.groupby('horizon_day')['log_err_sq'].mean().apply(np.sqrt)
    print("RMSLE by forecast horizon day (1 to 16):")
    for d, score in horizon_err.items():
        print(f"  Day {d:2d}: RMSLE = {score:.4f}")

    # Save summary to json
    summary_data = {
        'windows': all_window_results,
        'mean_rmsle': float(mean_rmsle),
        'overall_rmsle': float(overall_rmsle),
        'family_rmsle': fam_err.to_dict(),
        'store_rmsle': {int(k): float(v) for k, v in store_err.items()},
        'horizon_rmsle': {int(k): float(v) for k, v in horizon_err.items()}
    }
    with open('data/validation_summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    print("\nValidation summary saved to data/validation_summary.json")

if __name__ == '__main__':
    run_multi_window_evaluation()

