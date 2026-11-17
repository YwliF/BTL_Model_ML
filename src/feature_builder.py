import numpy as np
import pandas as pd
from src.feature_engineering import build_calendar_features, merge_external_data, compute_historical_target_encodings

class FastFeatureBuilder:
    def __init__(self, train_df, stores_df, oil_df, holidays_df):
        self.stores_df = stores_df
        self.oil_df = oil_df
        self.holidays_df = holidays_df
        
        # Build complete sales pivot (dates x 1782 store-family pairs)
        full_dates = pd.date_range('2013-01-01', '2017-08-31', freq='D', name='date')
        self.all_dates = full_dates
        
        raw_pivot = train_df.pivot(index='date', columns=['store_nbr', 'family'], values='sales')
        self.sales_pivot = raw_pivot.reindex(full_dates, fill_value=0.0).fillna(0.0)
        self.sales_pivot.index.name = 'date'
        self.log_sales_pivot = np.log1p(self.sales_pivot)
        
        # Onpromotion pivot
        raw_promo = train_df.pivot(index='date', columns=['store_nbr', 'family'], values='onpromotion')
        self.promo_pivot = raw_promo.reindex(full_dates, fill_value=0.0).fillna(0.0)
        self.promo_pivot.index.name = 'date'
        
        # Fixed order of store-family columns
        self.columns = self.sales_pivot.columns
        self.n_pairs = len(self.columns)
        self.stores_arr = np.array([c[0] for c in self.columns])
        self.fams_arr = np.array([c[1] for c in self.columns])

    def update_test_promo(self, test_df):
        """Updates promotion pivot with test.csv promotion values."""
        test_promo = test_df.pivot(index='date', columns=['store_nbr', 'family'], values='onpromotion').fillna(0.0)
        for d in test_promo.index:
            if d in self.promo_pivot.index:
                self.promo_pivot.loc[d] = test_promo.loc[d]

    def build_direct_features_for_cutoff(self, cutoff_date, horizon_step, train_start_date='2016-01-01', target_df=None):
        """
        Ultra-fast vectorized feature builder for step h (1 to 16).
        Target date is cutoff_date + h days.
        Zero leakage: For step h, the latest known information is at t - h <= cutoff_date.
        """
        h = horizon_step
        cutoff_dt = pd.to_datetime(cutoff_date)
        eval_date = cutoff_dt + pd.Timedelta(days=h)
        
        # Shift historical log sales by h
        lag_h = self.log_sales_pivot.shift(h)
        lag_h1 = self.log_sales_pivot.shift(h + 1)
        lag_h2 = self.log_sales_pivot.shift(h + 2)
        lag_h3 = self.log_sales_pivot.shift(h + 3)
        lag_h7 = self.log_sales_pivot.shift(h + 7)
        lag_h14 = self.log_sales_pivot.shift(h + 14)
        lag_h21 = self.log_sales_pivot.shift(h + 21)
        lag_h28 = self.log_sales_pivot.shift(h + 28)
        lag_h56 = self.log_sales_pivot.shift(h + 56)
        lag_h91 = self.log_sales_pivot.shift(h + 91)
        lag_h364 = self.log_sales_pivot.shift(h + 364)
        
        # Rolling stats computed on shift(h)
        roll_mean_7 = lag_h.rolling(7, min_periods=1).mean()
        roll_std_7 = lag_h.rolling(7, min_periods=1).std().fillna(0)
        roll_mean_14 = lag_h.rolling(14, min_periods=1).mean()
        roll_std_14 = lag_h.rolling(14, min_periods=1).std().fillna(0)
        roll_mean_28 = lag_h.rolling(28, min_periods=1).mean()
        roll_std_28 = lag_h.rolling(28, min_periods=1).std().fillna(0)
        roll_mean_56 = lag_h.rolling(56, min_periods=1).mean()
        ewma_7 = lag_h.ewm(span=7, min_periods=1).mean()
        ewma_14 = lag_h.ewm(span=14, min_periods=1).mean()
        
        all_tr_dates = pd.date_range(train_start_date, cutoff_dt, freq='D', name='date')
        
        def extract_slice_vectorized(date_idx):
            n_dates = len(date_idx)
            grid_dates = np.repeat(date_idx, self.n_pairs)
            grid_stores = np.tile(self.stores_arr, n_dates)
            grid_fams = np.tile(self.fams_arr, n_dates)
            
            p_slice = self.promo_pivot.loc[date_idx].to_numpy(dtype=np.float32).ravel()
            rm7 = roll_mean_7.loc[date_idx].to_numpy(dtype=np.float32).ravel()
            rm28 = roll_mean_28.loc[date_idx].to_numpy(dtype=np.float32).ravel()
            
            df = pd.DataFrame({
                'date': grid_dates,
                'store_nbr': grid_stores,
                'family': grid_fams,
                'lag_h': lag_h.loc[date_idx].to_numpy(dtype=np.float32).ravel(),
                'lag_h1': lag_h1.loc[date_idx].to_numpy(dtype=np.float32).ravel(),
                'lag_h2': lag_h2.loc[date_idx].to_numpy(dtype=np.float32).ravel(),
                'lag_h3': lag_h3.loc[date_idx].to_numpy(dtype=np.float32).ravel(),
                'lag_h7': lag_h7.loc[date_idx].to_numpy(dtype=np.float32).ravel(),
                'lag_h14': lag_h14.loc[date_idx].to_numpy(dtype=np.float32).ravel(),
                'lag_h21': lag_h21.loc[date_idx].to_numpy(dtype=np.float32).ravel(),
                'lag_h28': lag_h28.loc[date_idx].to_numpy(dtype=np.float32).ravel(),
                'lag_h56': lag_h56.loc[date_idx].to_numpy(dtype=np.float32).ravel(),
                'lag_h91': lag_h91.loc[date_idx].to_numpy(dtype=np.float32).ravel(),
                'lag_h364': lag_h364.loc[date_idx].to_numpy(dtype=np.float32).ravel(),
                'roll_mean_7': rm7,
                'roll_std_7': roll_std_7.loc[date_idx].to_numpy(dtype=np.float32).ravel(),
                'roll_mean_14': roll_mean_14.loc[date_idx].to_numpy(dtype=np.float32).ravel(),
                'roll_std_14': roll_std_14.loc[date_idx].to_numpy(dtype=np.float32).ravel(),
                'roll_mean_28': rm28,
                'roll_std_28': roll_std_28.loc[date_idx].to_numpy(dtype=np.float32).ravel(),
                'roll_mean_56': roll_mean_56.loc[date_idx].to_numpy(dtype=np.float32).ravel(),
                'ewma_7': ewma_7.loc[date_idx].to_numpy(dtype=np.float32).ravel(),
                'ewma_14': ewma_14.loc[date_idx].to_numpy(dtype=np.float32).ravel(),
                'onpromotion': p_slice,
                'momentum_7_28': (rm7 / (rm28 + 1e-4)).astype(np.float32)
            })
            return df

        # Build train slice
        train_feat = extract_slice_vectorized(all_tr_dates)
        train_feat['target'] = self.log_sales_pivot.loc[all_tr_dates].to_numpy(dtype=np.float32).ravel()
        
        # Build eval slice (at eval_date)
        eval_feat = extract_slice_vectorized([eval_date])
        if target_df is not None:
            eval_feat = eval_feat.merge(target_df[['date', 'store_nbr', 'family', 'sales']], on=['date', 'store_nbr', 'family'], how='left')
            eval_feat['target'] = np.log1p(eval_feat['sales'].fillna(0))
            
        return train_feat, eval_feat

    def build_lag16_features_for_cutoff(self, cutoff_date, train_start_date='2016-01-01', eval_dates=None):
        """
        Ultra-fast vectorized feature builder for Horizon-Agnostic Model using fixed lags >= 16.
        """
        cutoff_dt = pd.to_datetime(cutoff_date)
        
        lag_cols = [16, 17, 18, 19, 20, 21, 28, 35, 42, 56, 84, 91, 182, 364]
        lags_dict = {l: self.log_sales_pivot.shift(l) for l in lag_cols}
        
        lag_16 = lags_dict[16]
        roll_mean_7 = lag_16.rolling(7, min_periods=1).mean()
        roll_std_7 = lag_16.rolling(7, min_periods=1).std().fillna(0)
        roll_mean_14 = lag_16.rolling(14, min_periods=1).mean()
        roll_std_14 = lag_16.rolling(14, min_periods=1).std().fillna(0)
        roll_mean_28 = lag_16.rolling(28, min_periods=1).mean()
        roll_std_28 = lag_16.rolling(28, min_periods=1).std().fillna(0)
        roll_mean_56 = lag_16.rolling(56, min_periods=1).mean()
        ewma_7 = lag_16.ewm(span=7, min_periods=1).mean()
        ewma_14 = lag_16.ewm(span=14, min_periods=1).mean()
        
        tr_dates = pd.date_range(train_start_date, cutoff_dt, freq='D', name='date')
        
        def extract_slice_vectorized(date_idx):
            n_dates = len(date_idx)
            grid_dates = np.repeat(date_idx, self.n_pairs)
            grid_stores = np.tile(self.stores_arr, n_dates)
            grid_fams = np.tile(self.fams_arr, n_dates)
            
            p_slice = self.promo_pivot.loc[date_idx].to_numpy(dtype=np.float32).ravel()
            rm7 = roll_mean_7.loc[date_idx].to_numpy(dtype=np.float32).ravel()
            rm28 = roll_mean_28.loc[date_idx].to_numpy(dtype=np.float32).ravel()
            
            data_dict = {
                'date': grid_dates,
                'store_nbr': grid_stores,
                'family': grid_fams,
                'roll_mean_7': rm7,
                'roll_std_7': roll_std_7.loc[date_idx].to_numpy(dtype=np.float32).ravel(),
                'roll_mean_14': roll_mean_14.loc[date_idx].to_numpy(dtype=np.float32).ravel(),
                'roll_std_14': roll_std_14.loc[date_idx].to_numpy(dtype=np.float32).ravel(),
                'roll_mean_28': rm28,
                'roll_std_28': roll_std_28.loc[date_idx].to_numpy(dtype=np.float32).ravel(),
                'roll_mean_56': roll_mean_56.loc[date_idx].to_numpy(dtype=np.float32).ravel(),
                'ewma_7': ewma_7.loc[date_idx].to_numpy(dtype=np.float32).ravel(),
                'ewma_14': ewma_14.loc[date_idx].to_numpy(dtype=np.float32).ravel(),
                'onpromotion': p_slice,
                'momentum_7_28': (rm7 / (rm28 + 1e-4)).astype(np.float32)
            }
            for l in lag_cols:
                data_dict[f'lag_{l}'] = lags_dict[l].loc[date_idx].to_numpy(dtype=np.float32).ravel()
                
            return pd.DataFrame(data_dict)

        train_feat = extract_slice_vectorized(tr_dates)
        train_feat['target'] = self.log_sales_pivot.loc[tr_dates].to_numpy(dtype=np.float32).ravel()
        
        eval_feat = None
        if eval_dates is not None:
            eval_feat = extract_slice_vectorized(eval_dates)
            
        return train_feat, eval_feat

    def enrich_features(self, df, cutoff_date, train_history_df):
        """Enriches feature dataframe with calendar, stores, oil, holidays, and target encodings."""
        df = build_calendar_features(df)
        df = merge_external_data(df, self.stores_df, self.oil_df, self.holidays_df)
        
        # Holiday type fillna
        if 'national_h_type' in df.columns:
            df['national_h_type'] = df['national_h_type'].fillna('None').astype('category')
        
        # Target encodings strictly prior to cutoff
        sf_stats, sfd_stats, fd_stats, zero_pairs_set = compute_historical_target_encodings(train_history_df)
        
        df = df.merge(sf_stats, on=['store_nbr', 'family'], how='left')
        df = df.merge(sfd_stats, on=['store_nbr', 'family', 'dayofweek'], how='left')
        df = df.merge(fd_stats, on=['family', 'dayofweek'], how='left')
        
        # Fallbacks
        df['sf_smoothed_mean'] = df['sf_smoothed_mean'].fillna(0).astype(np.float32)
        df['sf_std_sales'] = df['sf_std_sales'].fillna(0).astype(np.float32)
        df['sfd_smoothed_mean'] = df['sfd_smoothed_mean'].fillna(0).astype(np.float32)
        df['fd_smoothed_mean'] = df['fd_smoothed_mean'].fillna(0).astype(np.float32)
        
        # Zero sales pair flag
        pair_tuples = list(zip(df['store_nbr'], df['family']))
        df['is_zero_pair'] = [1 if p in zero_pairs_set else 0 for p in pair_tuples]
        df['is_zero_pair'] = df['is_zero_pair'].astype(np.int8)
        
        return df
