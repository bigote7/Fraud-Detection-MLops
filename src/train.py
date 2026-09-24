import sys, mlflow, mlflow.sklearn, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 300
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 12
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('tp1-decouverte')
mlflow.sklearn.autolog()
df = pd.read_csv('data/transactions.csv')
X, y = df.drop(columns='fraude'), df['fraude']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
with mlflow.start_run(run_name=f'rf-{n_estimators}-{max_depth}'):
    m = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
    random_state=42).fit(Xtr, ytr)
    f1 = f1_score(yte, m.predict(Xte))
    mlflow.log_metric('f1_test', f1)
    print('F1 =', round(f1, 4))
