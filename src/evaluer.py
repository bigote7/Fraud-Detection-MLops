import json, joblib, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, ConfusionMatrixDisplay

te = pd.read_csv('data/prepare/test.csv')
X, y = te.drop(columns='fraude'), te['fraude']
m = joblib.load('models/model.pkl')
pred = m.predict(X)
mets = {'f1': round(f1_score(y, pred), 4),
        'precision': round(precision_score(y, pred), 4),
        'recall': round(recall_score(y, pred), 4)}
json.dump(mets, open('reports/metrics.json', 'w'), indent=2)
ConfusionMatrixDisplay.from_predictions(y, pred)
plt.savefig('reports/confusion.png', dpi=110, bbox_inches='tight')
print('evaluation :', mets)
