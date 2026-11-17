import numpy as np
import pandas as pd
from src.data_loader import prepare_oil_data, prepare_holidays_data

def build_calendar_features(df):
    """Generates rich calendar, seasonal, and cyclic features."""
    df = df.copy()
    dates = pd.to_datetime(df['date'])
    
    df['year'] = dates.dt.year.astype(np.int16)
    df['month'] = dates.dt.month.astype(np.int8)
    df['day'] = dates.dt.day.astype(np.int8)
    df['dayofweek'] = dates.dt.dayofweek.astype(np.int8)
    df['dayofyear'] = dates.dt.dayofyear.astype(np.int16)
    df['weekofyear'] = dates.dt.isocalendar().week.astype(np.int8)
    df['quarter'] = dates.dt.quarter.astype(np.int8)
    
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(np.int8)
    df['is_month_start'] = dates.dt.is_month_start.astype(np.int8)
    df['is_month_end'] = dates.dt.is_month_end.astype(np.int8)
    
    # Quincena features (Ecuador wage paydays on 15th and last day of month)
    days_in_month = dates.dt.days_in_month
    dist_15 = (df['day'] - 15).abs()
    dist_end = (days_in_month - df['day']).abs()
    df['dist_quincena'] = np.minimum(dist_15, dist_end).astype(np.int8)
    df['is_quincena'] = (df['dist_quincena'] <= 1).astype(np.int8)
    
    # Cyclic encoding
    df['sin_dow'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0).astype(np.float32)
    df['cos_dow'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0).astype(np.float32)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12.0).astype(np.float32)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12.0).astype(np.float32)
    df['sin_doy'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25).astype(np.float32)
    df['cos_doy'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25).astype(np.float32)
    
    return df

def merge_external_data(df, stores, oil, holidays):
    """Merges store metadata, oil price dynamics, and locale-aware holidays."""
    df = df.copy()
    
    # Merge store metadata
    df = df.merge(stores, on='store_nbr', how='left')
    
    # Merge oil
    oil_df = prepare_oil_data(oil, start_date='2013-01-01', end_date='2017-08-31')
    df = df.merge(oil_df, on='date', how='left')
    
    # Merge holidays
    national, regional, local, work_days = prepare_holidays_data(holidays, stores)
    
    # 1. National holidays
    df = df.merge(national[['date', 'is_national_holiday', 'type']].rename(columns={'type': 'national_h_type'}), on='date', how='left')
    df['is_national_holiday'] = df['is_national_holiday'].fillna(0).astype(np.int8)
    
    # 2. Regional holidays
    reg_merge = regional[['date', 'locale_name', 'is_regional_holiday']].rename(columns={'locale_name': 'state'})
    df = df.merge(reg_merge, on=['date', 'state'], how='left')
    df['is_regional_holiday'] = df['is_regional_holiday'].fillna(0).astype(np.int8)
    
    # 3. Local holidays
    loc_merge = local[['date', 'locale_name', 'is_local_holiday']].rename(columns={'locale_name': 'city'})
    df = df.merge(loc_merge, on=['date', 'city'], how='left')
    df['is_local_holiday'] = df['is_local_holiday'].fillna(0).astype(np.int8)
    
    # 4. Work days override
    work_dates = set(work_days['date'].dt.strftime('%Y-%m-%d'))
    date_strs = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    is_work_day = date_strs.isin(work_dates)
    
    df['is_holiday'] = ((df['is_national_holiday'] == 1) | (df['is_regional_holiday'] == 1) | (df['is_local_holiday'] == 1)).astype(np.int8)
    df.loc[is_work_day, 'is_holiday'] = 0
    df.loc[is_work_day, 'is_national_holiday'] = 0
    df.loc[is_work_day, 'is_regional_holiday'] = 0
    df.loc[is_work_day, 'is_local_holiday'] = 0
    
    # Promotion aggregations
    df['has_promotion'] = (df['onpromotion'] > 0).astype(np.int8)
    
    # Daily promo totals across store and family
    store_daily_promo = df.groupby(['date', 'store_nbr'])['onpromotion'].transform('sum').astype(np.float32)
    family_daily_promo = df.groupby(['date', 'family'])['onpromotion'].transform('sum').astype(np.float32)
    df['store_promo_sum'] = store_daily_promo
    df['family_promo_sum'] = family_daily_promo
    df['promo_ratio_in_store'] = (df['onpromotion'] / (store_daily_promo + 1.0)).astype(np.float32)
    
    # Convert categories
    cat_cols = ['family', 'city', 'state', 'type', 'cluster']
    for c in cat_cols:
        df[c] = df[c].astype('category')
        
    return df

def compute_historical_target_encodings(train_history, prior_weight=20.0):
    """
    Computes smoothed target encodings (mean log1p sales) on historical training data
    strictly prior to the evaluation window to eliminate any target leakage.
    """
    global_mean = np.log1p(train_history['sales']).mean()
    
    # 1. Store-Family smoothed mean and std
    sf_stats = train_history.groupby(['store_nbr', 'family'], observed=False)['sales'].agg(
        n='count',
        mean_sales=lambda x: np.log1p(x).mean(),
        std_sales=lambda x: np.log1p(x).std()
    ).reset_index()
    sf_stats['sf_smoothed_mean'] = (sf_stats['n'] * sf_stats['mean_sales'] + prior_weight * global_mean) / (sf_stats['n'] + prior_weight)
    sf_stats['sf_std_sales'] = sf_stats['std_sales'].fillna(0)
    sf_stats = sf_stats[['store_nbr', 'family', 'sf_smoothed_mean', 'sf_std_sales']]
    
    # 2. Store-Family-DayOfWeek smoothed mean
    sfd_stats = train_history.groupby(['store_nbr', 'family', 'dayofweek'], observed=False)['sales'].agg(
        n='count',
        mean_sales=lambda x: np.log1p(x).mean()
    ).reset_index()
    sfd_stats['sfd_smoothed_mean'] = (sfd_stats['n'] * sfd_stats['mean_sales'] + 10.0 * global_mean) / (sfd_stats['n'] + 10.0)
    sfd_stats = sfd_stats[['store_nbr', 'family', 'dayofweek', 'sfd_smoothed_mean']]
    
    # 3. Family-DayOfWeek smoothed mean
    fd_stats = train_history.groupby(['family', 'dayofweek'], observed=False)['sales'].agg(
        n='count',
        mean_sales=lambda x: np.log1p(x).mean()
    ).reset_index()
    fd_stats['fd_smoothed_mean'] = (fd_stats['n'] * fd_stats['mean_sales'] + 10.0 * global_mean) / (fd_stats['n'] + 10.0)
    fd_stats = fd_stats[['family', 'dayofweek', 'fd_smoothed_mean']]

    # 4. Zero-sales pair indicator (pairs with 0 sales in recent history)
    recent_cutoff = train_history['date'].max() - pd.Timedelta(days=60)
    recent_history = train_history[train_history['date'] >= recent_cutoff]
    zero_pairs = recent_history.groupby(['store_nbr', 'family'], observed=False)['sales'].sum()
    zero_pairs_set = set(zero_pairs[zero_pairs == 0].index)
    
    return sf_stats, sfd_stats, fd_stats, zero_pairs_set

