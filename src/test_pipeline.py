import sys, os
sys.path.insert(0, os.path.abspath('.'))
import time
import pandas as pd
import numpy as np
from src.data_loader import load_raw_data
from src.feature_builder import FastFeatureBuilder

print("Loading raw data...")
t0 = time.time()
train, test, stores, oil, holidays, sample_sub = load_raw_data()
print(f"Loaded in {time.time()-t0:.2f}s")

print("Initializing FastFeatureBuilder...")
t0 = time.time()
fb = FastFeatureBuilder(train, stores, oil, holidays)
print(f"Initialized in {time.time()-t0:.2f}s")

# Test building direct features for horizon step 1
print("Building direct features for horizon step 1 (eval date = 2017-08-01, cutoff = 2017-07-31)...")
t0 = time.time()
tr_feat, ev_feat = fb.build_direct_features_for_cutoff('2017-07-31', horizon_step=1, train_start_date='2017-01-01')
print(f"Built in {time.time()-t0:.2f}s: tr_feat shape {tr_feat.shape}, ev_feat shape {ev_feat.shape}")

# Test enrich
print("Enriching features with calendar, stores, oil, holidays, target encodings...")
t0 = time.time()
tr_history = train[train['date'] <= '2017-07-31'].copy()
tr_history['dayofweek'] = tr_history['date'].dt.dayofweek
tr_enriched = fb.enrich_features(tr_feat, '2017-07-31', tr_history)
ev_enriched = fb.enrich_features(ev_feat, '2017-07-31', tr_history)
print(f"Enriched train shape: {tr_enriched.shape}, eval shape: {ev_enriched.shape} in {time.time()-t0:.2f}s")
print(f"Sample columns:\n{tr_enriched.columns.tolist()[:15]} ... {tr_enriched.columns.tolist()[-10:]}")
print("Check NaNs in train:", tr_enriched.isna().sum()[tr_enriched.isna().sum() > 0].to_dict())
print("Check NaNs in eval:", ev_enriched.isna().sum()[ev_enriched.isna().sum() > 0].to_dict())
