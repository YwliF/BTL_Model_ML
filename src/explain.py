import sys, os
sys.path.insert(0, os.path.abspath('.'))
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from src.data_loader import load_raw_data
from src.feature_builder import FastFeatureBuilder

def run_shap_analysis():
    print("Loading data for SHAP analysis...")
    train, test, stores, oil, holidays, sample_sub = load_raw_data()
    fb = FastFeatureBuilder(train, stores, oil, holidays)

    cutoff_date = '2017-07-30'
    eval_dates = pd.date_range('2017-07-31', '2017-08-15', freq='D')

    train_history = train[train['date'] <= cutoff_date].copy()
    train_history['dayofweek'] = train_history['date'].dt.dayofweek

    drop_cols = ['date', 'target', 'sales']
    cat_cols = ['store_nbr', 'family', 'city', 'state', 'type', 'cluster', 'national_h_type']

    print("Building features...")
    tr_feat, ev_feat = fb.build_lag16_features_for_cutoff(cutoff_date, train_start_date='2016-08-01', eval_dates=eval_dates)
    tr_enriched = fb.enrich_features(tr_feat, cutoff_date, train_history)
    ev_enriched = fb.enrich_features(ev_feat, cutoff_date, train_history)

    for c in cat_cols:
        if c in tr_enriched.columns:
            tr_enriched[c] = tr_enriched[c].astype('category')
            ev_enriched[c] = ev_enriched[c].astype('category')

    feature_cols = [c for c in tr_enriched.columns if c not in drop_cols]
    
    # Load tuned parameters if available
    param_path = 'data/best_hyperparams.json'
    if os.path.exists(param_path):
        with open(param_path) as f:
            params = json.load(f)
    else:
        params = {
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
    params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    })

    print("Fitting model for SHAP analysis...")
    model = lgb.LGBMRegressor(**params)
    model.fit(tr_enriched[feature_cols], tr_enriched['target'])

    # Feature Importance (Gain & Split)
    imp_gain = pd.Series(model.booster_.feature_importance(importance_type='gain'), index=feature_cols).sort_values(ascending=False)
    imp_split = pd.Series(model.booster_.feature_importance(importance_type='split'), index=feature_cols).sort_values(ascending=False)

    print("\n=================== TOP 20 FEATURES BY LIGHTGBM GAIN ===================")
    for i, (feat, val) in enumerate(imp_gain.head(20).items(), 1):
        print(f"  {i:2d}. {feat:<25} Gain: {val:14,.1f} | Splits: {imp_split[feat]:6,d}")

    # SHAP Analysis with TreeExplainer
    print("\nRunning TreeExplainer on validation sample (5,000 points across stores and families)...")
    np.random.seed(42)
    sample_indices = np.random.choice(len(ev_enriched), size=min(5000, len(ev_enriched)), replace=False)
    X_sample = ev_enriched[feature_cols].iloc[sample_indices]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.Series(mean_abs_shap, index=feature_cols).sort_values(ascending=False)

    print("\n=================== TOP 20 FEATURES BY MEAN ABSOLUTE SHAP ===================")
    for i, (feat, val) in enumerate(shap_importance.head(20).items(), 1):
        print(f"  {i:2d}. {feat:<25} Mean |SHAP|: {val:.4f}")

    # Save plots and artifacts
    os.makedirs('reports', exist_ok=True)

    # 1. Plot Top 20 SHAP
    fig, ax = plt.subplots(figsize=(10, 8))
    top20_shap = shap_importance.head(20).sort_values(ascending=True)
    y_pos = np.arange(len(top20_shap))
    ax.barh(y_pos, top20_shap.values, color='#1f77b4', edgecolor='black', alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top20_shap.index, fontsize=10)
    ax.set_xlabel('Mean |SHAP Value| (Impact on Log Sales Prediction)', fontsize=11, fontweight='bold')
    ax.set_title('Top 20 Features by SHAP Importance (Store Sales LightGBM)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/shap_top20_importance.png', dpi=300)
    plt.close()
    print("Saved plot to reports/shap_top20_importance.png")

    # 2. Save json
    top20_dict = {
        'shap_importance': {feat: float(val) for feat, val in shap_importance.head(20).items()},
        'gain_importance': {feat: float(val) for feat, val in imp_gain.head(20).items()}
    }
    with open('reports/top20_features.json', 'w') as f:
        json.dump(top20_dict, f, indent=2)
    print("Saved json to reports/top20_features.json")

if __name__ == '__main__':
    run_shap_analysis()

