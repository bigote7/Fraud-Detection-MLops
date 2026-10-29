import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
df = pd.read_csv('data/transactions.csv')
X, y = df.drop(columns='fraude'), df['fraude']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
m = LogisticRegression(max_iter=1000).fit(Xtr, ytr) # modele de reference
print('reference F1 =', round(f1_score(yte, m.predict(Xte)), 4))
