import pandas as pd

df = pd.read_csv(r'C:/Users/akrit/Downloads/mlops project/G200-MP236-MLOps/data/raw/train.csv')
# ensure date parsed
df['date'] = pd.to_datetime(df['date'])
mask = (df['vehicle_age_yr'] == 0.9) & (df['date'] >= pd.Timestamp('2026-02-15')) & (df['date'] <= pd.Timestamp('2026-03-03'))
print('masked count', mask.sum())
print('total 0.9 count', (df['vehicle_age_yr']==0.9).sum())
missing = df[(df['vehicle_age_yr']==0.9) & ~mask]
print('rows not masked')
print(missing[['vehicle_age_yr','date']].head(20))
