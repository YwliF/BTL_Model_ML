import pandas as pd
import numpy as np

def load_raw_data(data_dir='data'):
    """Loads all competition raw CSV files."""
    train = pd.read_csv(f'{data_dir}/train.csv', parse_dates=['date'])
    test = pd.read_csv(f'{data_dir}/test.csv', parse_dates=['date'])
    stores = pd.read_csv(f'{data_dir}/stores.csv')
    oil = pd.read_csv(f'{data_dir}/oil.csv', parse_dates=['date'])
    holidays = pd.read_csv(f'{data_dir}/holidays_events.csv', parse_dates=['date'])
    sample_sub = pd.read_csv(f'{data_dir}/sample_submission.csv')
    return train, test, stores, oil, holidays, sample_sub

def prepare_oil_data(oil, start_date='2013-01-01', end_date='2017-08-31'):
    """Creates a continuous daily oil price series with interpolation and momentum features."""
    all_dates = pd.date_range(start_date, end_date, freq='D')
    oil_df = pd.DataFrame({'date': all_dates})
    oil_df = oil_df.merge(oil, on='date', how='left')
    
    # Linear interpolation for weekends & missing holidays, then bfill and ffill
    oil_df['dcoilwtico'] = oil_df['dcoilwtico'].interpolate(method='linear')
    oil_df['dcoilwtico'] = oil_df['dcoilwtico'].bfill().ffill()
    
    # Feature engineering for oil
    oil_df['oil_rolling_mean_7'] = oil_df['dcoilwtico'].rolling(7, min_periods=1).mean()
    oil_df['oil_rolling_mean_14'] = oil_df['dcoilwtico'].rolling(14, min_periods=1).mean()
    oil_df['oil_rolling_mean_30'] = oil_df['dcoilwtico'].rolling(30, min_periods=1).mean()
    oil_df['oil_diff_1'] = oil_df['dcoilwtico'].diff(1).fillna(0)
    oil_df['oil_diff_7'] = oil_df['dcoilwtico'].diff(7).fillna(0)
    return oil_df

def prepare_holidays_data(holidays, stores):
    """
    Parses holidays considering transfer status and locales (National, Regional, Local).
    Returns a dataframe indexed by (date, store_nbr) with holiday indicators.
    """
    # 1. Filter out transferred holidays (transferred == True are normal work days)
    # The actual holiday occurred on the day with type == 'Transfer'
    valid_holidays = holidays[holidays['transferred'] == False].copy()
    
    # Distinguish work days (which undo holidays)
    work_days = valid_holidays[valid_holidays['type'] == 'Work Day']
    actual_holidays = valid_holidays[valid_holidays['type'] != 'Work Day']
    
    # Map holidays to stores based on locale
    # National holidays apply to all 54 stores
    national = actual_holidays[actual_holidays['locale'] == 'National'][['date', 'type', 'description']].drop_duplicates('date')
    national['is_national_holiday'] = 1
    
    # Regional holidays apply to stores in that state
    regional = actual_holidays[actual_holidays['locale'] == 'Regional'][['date', 'locale_name', 'type']].drop_duplicates(['date', 'locale_name'])
    regional['is_regional_holiday'] = 1
    
    # Local holidays apply to stores in that city
    local = actual_holidays[actual_holidays['locale'] == 'Local'][['date', 'locale_name', 'type']].drop_duplicates(['date', 'locale_name'])
    local['is_local_holiday'] = 1

    return national, regional, local, work_days

