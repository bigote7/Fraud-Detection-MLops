import yaml, os, pandas as pd
from sklearn.model_selection import train_test_split

p = yaml.safe_load(open('params.yaml'))['preparation']
df = pd.read_csv('data/transactions.csv')
tr, te = train_test_split(df, test_size=p['test_size'], random_state=p['random_state'])
os.makedirs('data/prepare', exist_ok=True)
tr.to_csv('data/prepare/train.csv', index=False)
te.to_csv('data/prepare/test.csv', index=False)
print('preparation :', len(tr), 'lignes train /', len(te), 'lignes test')
