# Fraud Detection MLOps

Détection de fraude sur des transactions synthétiques, avec suivi des expériences MLflow et versionnement des données DVC.

Un classifieur Random Forest prédit si une transaction est frauduleuse à partir du montant, de l’ancienneté du compte, de la fréquence sur 24 h et d’un indicateur de transaction nocturne.

## Structure

| TP | Sujet | Fichiers |
|----|-------|----------|
| TP 1 | Suivi des expériences MLflow | `donnees.py`, `src/train.py`, `run_tp1.ps1` |
| TP 2 | Pipeline reproductible | `pipeline.py`, `MLproject`, `params.yaml`, `src/preparer.py`, `src/entrainer.py`, `src/evaluer.py` |
| TP 3 | Registre de modèles | `src/train_complet.py` |
| TP 4 | Versionnement des données et pipeline DVC | `data/transactions.csv.dvc`, `dvc.yaml`, `dvc.lock`, `reports/metrics.json` |
| TP 5 | AutoML (FLAML) et low-code (Orange, KNIME) | `src/reference.py`, `src/automl.py`, `workflows/fraude-tp5.ows`, `workflows/tp5-fraude.knwf`, `reports/comparatif.md` |

Les captures d'écran sont dans `photo/`.

## TP 5

```powershell
python src/reference.py      # régression logistique de référence
python src/automl.py 60      # FLAML, budget en secondes, journalisé dans MLflow
```

- Orange : ouvrir `workflows/fraude-tp5.ows` (File > Open).
- KNIME : Menu > Import, choisir `workflows/tp5-fraude.knwf`, puis vérifier le chemin du CSV Reader avant Execute all.
