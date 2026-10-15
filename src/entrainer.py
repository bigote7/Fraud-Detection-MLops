import yaml, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
p = yaml.safe_load(open('params.yaml'))['entrainement']
tr = pd.read_csv('data/prepare/train.csv')
X, y = tr.drop(columns='fraude'), tr['fraude']
m = RandomForestClassifier(n_estimators=p['n_estimators'],
                           max_depth=p['max_depth'], random_state=42).fit(X, y)
joblib.dump(m, 'models/model.pkl')
print('entrainement : modele ecrit dans models/model.pkl')
