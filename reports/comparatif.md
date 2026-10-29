# TP 5 - Comparatif AutoML et low-code

Jeu de test : 25 % de `data/transactions.csv` (1 250 lignes, 142 fraudes). Metrique : F1 de la classe 1 (fraude).

| Approche                    | F1 cl. 1 | Temps humain | Temps machine | Cout  | Explicabilite | Export          |
|-----------------------------|----------|--------------|---------------|-------|---------------|-----------------|
| Pipeline DVC (TP 3 / TP 4)  | 0.8593   | ~2 h         | 5 min         | 0 MAD | totale        | MLflow          |
| FLAML 60 s                  | a mesurer| 5 min        | 1 min         | 0 MAD | bonne         | MLflow          |
| Orange, Gradient Boosting   | 0.837    | ?            | ?             | 0 MAD | ?             | .pkcls          |
| KNIME, meilleur modele      | 0.494    | ?            | ?             | 0 MAD | ?             | .knwf           |
| KNIME, composant AutoML     | a mesurer| ?            | ?             | 0 MAD | ?             | workflow KNIME  |

## Orange (`workflows/fraude-tp5.ows`)

Test and Score, Test on test data, cible 1, avant amelioration :

| Modele              | AUC   | CA    | F1    | Prec  | Recall |
|---------------------|-------|-------|-------|-------|--------|
| Tree                | 0.845 | 0.950 | 0.756 | 0.857 | 0.676  |
| Logistic Regression | 0.979 | 0.952 | 0.781 | 0.811 | 0.754  |
| kNN                 | 0.819 | 0.906 | 0.491 | 0.633 | 0.401  |
| Random Forest       | 0.977 | 0.958 | 0.813 | 0.831 | 0.796  |
| Gradient Boosting   | 0.989 | 0.965 | 0.837 | 0.883 | 0.796  |

Apres amelioration (kNN normalise, foret a 100 arbres) : a mesurer.

## KNIME (`workflows/tp5-fraude.knwf`)

Matrices des trois Scorers sur les 1 250 lignes de test :

| Modele                 | TN   | FP | FN  | TP | F1    | Prec  | Recall |
|------------------------|------|----|-----|----|-------|-------|--------|
| Logistic Regression    | 1052 | 56 | 77  | 65 | 0.494 | 0.537 | 0.458  |
| Random Forest          | 1108 | 0  | 137 | 5  | 0.068 | 1.000 | 0.035  |
| Gradient Boosted Trees | 1097 | 11 | 94  | 48 | 0.478 | 0.814 | 0.338  |

Ces F1 sont tres inferieurs a ceux d'Orange et de scikit-learn sur le meme fichier
(regression logistique 0.70, foret 0.79, gradient boosting 0.82 avec le meme partage stratifie 75/25, graine 42) :
execution KNIME a refaire et a verifier avant de conclure.

## Conclusion

A rediger : dans quel contexte recommander chacune des approches ?
