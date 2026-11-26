import pandas as pd
df = pd.read_csv(r'C:/Users/akrit/Downloads/mlops project/G200-MP236-MLOps/data/raw/train.csv')
mask = df['vehicle_age_yr'] == 0.9
print('total 0.9 rows', mask.sum())
print(df[mask][['vehicle_age_yr','date']].head(20))