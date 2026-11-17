import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv', parse_dates=['date'])

train['year'] = train['date'].dt.year
print("Promo missing / zero rate by year:")
for yr, df_yr in train.groupby('year'):
    promo_zeros = (df_yr['onpromotion'] == 0).mean()
    promo_nans = df_yr['onpromotion'].isna().mean()
    promo_pos = (df_yr['onpromotion'] > 0).mean()
    print(f"Year {yr}: promo > 0: {promo_pos:.4f}, promo == 0: {promo_zeros:.4f}, NaNs: {promo_nans:.4f}")

# Check zero sales by store-family
st_fam_total = train.groupby(['store_nbr', 'family'])['sales'].sum()
zero_pairs = st_fam_total[st_fam_total == 0]
print(f"\nStore-Family pairs with total sales == 0 across all history: {len(zero_pairs)}")
if len(zero_pairs) > 0:
    print(zero_pairs)

# Check zero sales in recent history (e.g. 2017)
st_fam_2017 = train[train.date >= '2017-01-01'].groupby(['store_nbr', 'family'])['sales'].sum()
zero_pairs_2017 = st_fam_2017[st_fam_2017 == 0]
print(f"Store-Family pairs with total sales == 0 in 2017: {len(zero_pairs_2017)}")
if len(zero_pairs_2017) > 0:
    print("Pairs with 0 sales in 2017:", zero_pairs_2017.index.tolist())

