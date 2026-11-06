# SignLab — Classification de panneaux

Projet annuel IABD ESGI de Clyde Kianmene Takuefou. SignLab permet d’entraîner, comparer et interroger **23 classifieurs Python écrits avec NumPy** sur les mêmes partitions de photos : **STOP**, **sens interdit** et **limitation à 50 km/h**. Quatre modèles C++ accessibles depuis Python complètent le parcours prévu au syllabus.

Les trois dossiers de données sont vides, prêts à recevoir tes images. Les résultats de vérification fournis concernent des données synthétiques séparées.

## Démarrer sous Windows

Depuis PowerShell à la racine du projet avec Python 3.12 :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-verified.txt
.\.venv\Scripts\python.exe -m signlab catalog
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Sous macOS ou Linux, utilise `.venv/bin/python`. La version de Python et des dépendances réellement utilisée figure dans `environment.json` pour chaque expérience. Le projet fonctionne depuis sa racine, sans installation de SignLab. Pour une installation éditable avec Jupyter : `python -m pip install -e ".[notebook]"`.

## Ajouter tes photos

| Dossier | Classe |
| --- | --- |
| `data/images/stop/` | STOP |
| `data/images/sens_interdit/` | Sens interdit |
| `data/images/limitation_50/` | Limitation à 50 km/h |

Mets des photos cadrées du panneau attendu. Il faut au moins trois groupes indépendants par classe pour remplir entraînement, validation et test. Des photos répétées du même panneau ou de la même scène doivent partager un identifiant de groupe. Le guide [Données et protocole](docs/PROTOCOLE.md) décrit le CSV facultatif pour renseigner groupes, sources et licences. Le chargeur signale les fichiers illisibles, retire les doublons exacts et regroupe des quasi-doublons probables.

## Entraîner et explorer

Une fois les dossiers remplis :

```powershell
python -m signlab audit-data
python -m signlab benchmark --output runs/premier-benchmark --quick
python -m signlab serve --run runs/premier-benchmark
```

Ouvre [SignLab sur ton ordinateur](http://127.0.0.1:8000). Choisis un modèle, charge une photo et compare les scores de validation. Tu peux aussi lancer `python -m signlab serve` dès maintenant pour consulter le catalogue.

Le mode `--quick` utilise un réglage et une graine par modèle. Retire ce drapeau pour tester les réglages et graines du catalogue. Le réglage par défaut utilise jusqu’à **2 500 images d’entraînement**, choisies par groupes pour tous les modèles. Validation et test gardent leur taille d’origine. Pour exploiter tout l’entraînement sur les modèles sans noyau, copie la configuration, mets `train_limit` à `null` et sélectionne les modèles concernés.

Pour isoler une famille, étudier les courbes et produire une prédiction :

```powershell
python -m signlab benchmark --models majority logistic linear_svm mlp --output runs/comparaison-lineaire
python -m signlab diagnose --model mlp --output runs/diagnostic-mlp
python -m signlab predict --artifact runs/comparaison-lineaire/models/mlp.json --image data/images/stop/ma-photo.jpg
```

Les réglages se choisissent sur la validation. Une fois le protocole figé, `python -m signlab finalize --run runs/premier-benchmark --all-models` évalue les modèles sauvegardés sur le test. Omet `--all-models` pour évaluer uniquement le modèle retenu. Les métriques, matrices de confusion, erreurs, hyperparamètres, partitions, durées et artefacts se trouvent dans le dossier de l’expérience.

## Vérifier sans photos

```powershell
python scripts/synthetic_demo.py --output runs/fixture
python -m signlab benchmark --config runs/fixture/config.json --output runs/demo --quick
python -m signlab serve --run runs/demo
```

Les images générées servent à vérifier le parcours technique. Les répertoires de données réelles restent disponibles pour tes propres photos. Chaque expérience requiert un nouveau dossier de sortie.

## Compiler les quatre modèles C++

Ouvre le terminal développeur de Visual Studio avec les outils C++ installés, ou utilise un terminal qui dispose de GCC ou Clang. Utilise un compilateur de même architecture que Python.

```powershell
python scripts/build_native.py
python -m unittest tests.test_native -v
python -m signlab benchmark --models native_linear native_svm native_mlp native_rbf --output runs/natif --quick
```

`SIGNLAB_NATIVE_LIBRARY` permet de définir le chemin vers une bibliothèque compilée ailleurs. La liaison passe par [l’interface ctypes de Python](https://docs.python.org/3/library/ctypes.html).

## Étudier le projet

- [Guide des 23 modèles](docs/ALGORITHMES.md) — principes, hyperparamètres, faiblesses et exercices.
- [Données et protocole](docs/PROTOCOLE.md) — groupes, biais, métriques et comparaison des résolutions.
- [Audit technique](docs/AUDIT.md) — défauts confirmés, remédiations et limites restantes.
- [Notebook pédagogique](notebooks/signlab.ipynb) — exploration, entraînement, lecture des erreurs et cas XOR.
- [Rapport technique et expérimental](docs/RAPPORT.md) — démarche et état des validations.
- [Mesures de vérification synthétique](docs/verification-synthetique.json) — résultats enregistrés des 120 essais et du test final technique.
- [Code des modèles Python](signlab/models) et [bibliothèque C++](native/signlab.cpp).

Pour régénérer la version PDF du rapport à partir des mesures enregistrées :

```powershell
python -m pip install -e ".[report]"
python scripts/build_report.py
```

Les anciens modèles, journaux et notes sont conservés comme archives. Les nouveaux artefacts enregistrent le prétraitement, l’ordre des classes et l’empreinte du dataset dans un format JSON versionné. L’application locale classe une image dans les trois catégories connues. Elle ne localise pas les panneaux et ses scores ne sont pas calibrés.
