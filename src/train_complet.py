import sys, json, subprocess, mlflow, mlflow.sklearn, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, ConfusionMatrixDisplay)

n_estimators = int(sys.argv[1]); max_depth = int(sys.argv[2])
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('tp3-registre')
df = pd.read_csv('data/transactions.csv')
X, y = df.drop(columns='fraude'), df['fraude']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
with mlflow.start_run(run_name=f'rf-{n_estimators}-{max_depth}'):
    m = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                               random_state=42).fit(Xtr, ytr)
    pred = m.predict(Xte); proba = m.predict_proba(Xte)[:, 1]
    mlflow.log_params({'n_estimators': n_estimators, 'max_depth': max_depth})
    mlflow.log_metrics({'f1': f1_score(yte, pred),
                        'precision': precision_score(yte, pred),
                        'recall': recall_score(yte, pred),
                        'roc_auc': roc_auc_score(yte, proba)})
    commit = subprocess.getoutput('git rev-parse --short HEAD')
    mlflow.set_tags({'auteur': 'LABIB LAYACHI', 'commit': commit,
                     'jeu_de_donnees': 'transactions v1'})
    ConfusionMatrixDisplay.from_predictions(yte, pred)
    plt.savefig('reports/confusion.png', dpi=110, bbox_inches='tight')
    mlflow.log_artifact('reports/confusion.png')
    imp = dict(zip(X.columns, m.feature_importances_.round(4)))
    json.dump(imp, open('reports/importances.json', 'w'), indent=2)
    mlflow.log_artifact('reports/importances.json')
    mlflow.sklearn.log_model(m, name='model', input_example=Xte[:3],
                             skops_trusted_types=['sklearn.tree._tree.Tree'])
    print('F1 =', round(f1_score(yte, pred), 4))
