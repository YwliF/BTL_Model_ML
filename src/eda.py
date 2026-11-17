import os
import pandas as pd
import numpy as np

def run_eda():
    train = pd.read_csv('data/train.csv', parse_dates=['date'])
    test = pd.read_csv('data/test.csv', parse_dates=['date'])
    stores = pd.read_csv('data/stores.csv')
    oil = pd.read_csv('data/oil.csv', parse_dates=['date'])
    holidays = pd.read_csv('data/holidays_events.csv', parse_dates=['date'])
    trans = pd.read_csv('data/transactions.csv', parse_dates=['date'])

    print("=================== 1. DATA OVERVIEW ===================")
    print(f"Train rows: {len(train):,}, Date range: {train.date.min().date()} -> {train.date.max().date()} ({train.date.nunique()} unique days)")
    print(f"Test rows:  {len(test):,}, Date range: {test.date.min().date()} -> {test.date.max().date()} ({test.date.nunique()} unique days)")
    print(f"Number of stores: {stores.store_nbr.nunique()}, Families: {train.family.nunique()}")
    
    n_pairs_train = train.groupby(['store_nbr', 'family']).ngroups
    n_pairs_test = test.groupby(['store_nbr', 'family']).ngroups
    print(f"Store-Family pairs: {n_pairs_train} in train, {n_pairs_test} in test (54 * 33 = {54*33})")
    
    # Check completeness of calendar dates
    full_date_range = pd.date_range(train.date.min(), train.date.max())
    missing_dates = sorted(list(set(full_date_range) - set(train.date.unique())))
    print(f"Missing calendar dates in train: {[d.strftime('%Y-%m-%d') for d in missing_dates]}")
    print("Notice: December 25 (Christmas) is missing every year because stores are closed!")

    print("\n=================== 2. SALES DISTRIBUTION ===================")
    print(f"Sales Min: {train.sales.min()}, Max: {train.sales.max():,.2f}")
    print(f"Sales Mean: {train.sales.mean():.2f}, Median: {train.sales.median():.2f}, Std: {train.sales.std():.2f}")
    zero_sales = (train.sales == 0).sum()
    print(f"Zero sales count: {zero_sales:,} ({zero_sales / len(train) * 100:.2f}%)")
    log_sales = np.log1p(train.sales)
    print(f"Log1p(Sales) Mean: {log_sales.mean():.4f}, Std: {log_sales.std():.4f}, Skew: {log_sales.skew():.4f}")

    print("\n=================== 3. FAMILY-LEVEL ANALYSIS ===================")
    fam_stats = train.groupby('family')['sales'].agg(['count', 'mean', 'sum', lambda x: (x == 0).mean()]).rename(columns={'<lambda_0>': 'zero_pct'})
    fam_stats['sales_share_%'] = (fam_stats['sum'] / train['sales'].sum()) * 100
    fam_stats = fam_stats.sort_values(by='sum', ascending=False)
    print("Top 10 highest-selling families:")
    print(fam_stats[['sum', 'sales_share_%', 'mean', 'zero_pct']].head(10).to_string())
    print("\nBottom 5 lowest-selling families:")
    print(fam_stats[['sum', 'sales_share_%', 'mean', 'zero_pct']].tail(5).to_string())

    print("\n=================== 4. STORE-LEVEL ANALYSIS ===================")
    store_sales = train.groupby('store_nbr')['sales'].agg(['sum', 'mean']).sort_values('sum', ascending=False)
    print(f"Top 5 highest sales stores: {store_sales.head(5).index.tolist()}")
    print(f"Bottom 5 lowest sales stores: {store_sales.tail(5).index.tolist()}")
    
    # Check when each store opened (first non-zero sales date or first recorded date)
    store_first_dates = train[train.sales > 0].groupby('store_nbr')['date'].min()
    late_opening = store_first_dates[store_first_dates > train.date.min()]
    print(f"\nStores opened after 2013-01-01:")
    for s_id, d in late_opening.items():
        print(f"  Store {s_id}: first sale on {d.strftime('%Y-%m-%d')}")

    print("\n=================== 5. PROMOTION IMPACT ===================")
    promo_sales = train.groupby(train['onpromotion'] > 0)['sales'].agg(['count', 'mean', 'median'])
    print("Sales with vs without promotion:")
    print(promo_sales.to_string())
    promo_missing_test = test['onpromotion'].isna().sum()
    print(f"Test onpromotion NaNs: {promo_missing_test}")

    print("\n=================== 6. OIL PRICE ANALYSIS ===================")
    print(f"Oil date range: {oil.date.min().date()} to {oil.date.max().date()}")
    print(f"Oil rows: {len(oil)}, Missing dcoilwtico: {oil.dcoilwtico.isna().sum()} ({oil.dcoilwtico.isna().mean()*100:.2f}%)")
    
    # Check oil dates coverage against train + test
    full_dates = pd.date_range(train.date.min(), test.date.max())
    oil_indexed = oil.set_index('date').reindex(full_dates)
    print(f"Total days from train start to test end: {len(full_dates)}")
    print(f"Days missing oil before interpolation (weekends + holidays): {oil_indexed['dcoilwtico'].isna().sum()}")

    print("\n=================== 7. HOLIDAYS & EVENTS ===================")
    print(f"Holidays total rows: {len(holidays)}")
    print("Holiday locale distribution:\n", holidays['locale'].value_counts().to_string())
    print("Holiday type distribution:\n", holidays['type'].value_counts().to_string())
    print(f"Transferred holidays count: {(holidays['transferred'] == True).sum()}")
    
    # Earthquake impact check (2016-04-16)
    print("\n=================== 8. ANOMALIES & EARTHQUAKE ===================")
    eq_start = pd.Timestamp('2016-04-16')
    eq_window = train[(train.date >= '2016-04-10') & (train.date <= '2016-05-15')]
    eq_daily = eq_window.groupby('date')['sales'].sum()
    print("Daily sales around 7.8 Mw Earthquake on 2016-04-16:")
    print(eq_daily.head(10).to_string())

    print("\n=================== 9. TRANSACTIONS ===================")
    print(f"Transactions rows: {len(trans)}, Date range: {trans.date.min().date()} to {trans.date.max().date()}")
    print("Important: Transactions are NOT available for the test period (2017-08-16 to 2017-08-31)!")
    print("Therefore, transactions cannot be directly used as a feature at test time unless predicted or omitted.")

if __name__ == '__main__':
    run_eda()

