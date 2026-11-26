import pandas as pd
import sys
df = pd.read_csv(r'C:/Users/akrit/Downloads/mlops project/G200-MP236-MLOps/data/raw/train.csv')
mask = (df['vehicle_age_yr'] == 0.9) & (df['date'] >= '2026-02-15') & (df['date'] <= '2026-03-03')
print('matches', mask.sum())
print(df[mask])
