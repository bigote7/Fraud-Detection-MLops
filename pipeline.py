import subprocess, json, yaml, joblib, mlflow, mlflow.sklearn

P = yaml.safe_load(open('params.yaml'))
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('tp2-pipeline')

with mlflow.start_run(run_name='pipeline'):
    for etape in ['preparer', 'entrainer', 'evaluer']:
        with mlflow.start_run(run_name=etape, nested=True):
            subprocess.run(['python', f'src/{etape}.py'], check=True)

    mets = json.load(open('reports/metrics.json'))
    mlflow.log_params({**P['preparation'], **P['entrainement']})
    mlflow.log_metrics(mets)
    mlflow.log_artifact('reports/confusion.png')

    seuil = P['validation']['seuil_f1']
    if mets['f1'] >= seuil:
        mlflow.sklearn.log_model(joblib.load('models/model.pkl'), name='model',
                                 registered_model_name='fraude-tp2',
                                 skops_trusted_types=['sklearn.tree._tree.Tree'])
        print(f"modele accepte (F1 {mets['f1']} >= {seuil}) et enregistre")
    else:
        mlflow.set_tag('statut', 'refuse')
        raise SystemExit(f"modele refuse : F1 {mets['f1']} < seuil {seuil}")
