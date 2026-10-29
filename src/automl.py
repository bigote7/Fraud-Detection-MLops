import sys, time, pandas as pd, mlflow
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
budget = int(sys.argv[1]) if len(sys.argv) > 1 else 60 # secondes de calcul
df = pd.read_csv('data/transactions.csv')
X, y = df.drop(columns='fraude'), df['fraude']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
t0 = time.time()
a = AutoML()
a.fit(Xtr, ytr, task='classification', metric='f1',
      time_budget=budget, verbose=0) # la recherche
duree = round(time.time() - t0, 1)
f1 = f1_score(yte, a.predict(Xte))
print('budget', budget, 's | modele', a.best_estimator, '| F1', round(f1, 4))
print('config :', a.best_config)
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('tp5-automl')
with mlflow.start_run(run_name=f'flaml-{budget}s'): # on trace la recherche
    mlflow.log_params({'outil': 'flaml', 'budget_s': budget,
                       'modele_retenu': a.best_estimator})
    mlflow.log_metrics({'f1': f1, 'duree_s': duree})
