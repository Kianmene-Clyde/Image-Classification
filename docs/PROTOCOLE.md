# Données et protocole

## Définir les unités indépendantes

Une classe doit désigner un panneau identifiable sur une photo cadrée. Une séquence vidéo, une rafale ou des versions recadrées de la même photo constituent un groupe. Utilise un identifiant commun même si deux fichiers ont des tailles différentes. Pour tester un changement de domaine, utilise par exemple l’identifiant d’un lieu, d’une caméra ou d’une source comme unité de séparation.

Le CSV de métadonnées se configure via `metadata_csv`, avec un chemin relatif au fichier de configuration. Exemple :

```csv
relative_path,group,source,license
stop/photo_01.jpg,scene-001,photo-personnelle,droits-auteur
stop/photo_02.jpg,scene-001,photo-personnelle,droits-auteur
limitation_50/photo_03.jpg,scene-011,photo-personnelle,droits-auteur
```

L’audit contrôle le décodage, convertit les couleurs en RGB, applique l’orientation EXIF et repère les doublons de pixels avec SHA-256. Les quasi-doublons candidats combinent une empreinte dHash et une distance de couleur sur une miniature. Cette heuristique ne repère pas toutes les variantes d’une même scène. Les groupes renseignés manuellement apportent cette information. Un doublon exact portant deux classes interrompt le chargement pour permettre une correction d’étiquette.

## Évaluer avec les mêmes données

1. L’audit fige la liste des fichiers, les groupes et l’empreinte du dataset.
2. La partition groupée approche les proportions par classe sur trois ensembles. Inspecte les effectifs et la diversité de provenance avant de comparer.
3. Si `train_limit` est défini, une sous-cohorte groupée commune est tirée au sein du train. L’index effectif est conservé dans `training_indices.json`.
4. La PCA, la standardisation et les bornes min–max sont calculées sur cet entraînement uniquement. Les modèles reçoivent la même représentation de départ, avec une mise à l’échelle adaptée à leurs hypothèses.
5. Les candidats sont classés sur la moyenne du macro-F1 de validation sur les graines fixées. Le premier réglage départage les égalités à précision numérique. L’artefact conservé utilise la première graine déclarée.
6. Après avoir choisi la représentation, les modèles et les hyperparamètres, évalue le test une fois. Pour effectuer une nouvelle optimisation ensuite, réserve un nouveau test indépendant.

Le macro-F1 donne le même poids à chaque classe, quelle que soit sa fréquence. La précision mesure la proportion de vraies observations parmi les prédictions d’une classe. Le rappel mesure la proportion d’observations de cette classe correctement retrouvées. La matrice de confusion place les vraies classes en lignes et les prédictions en colonnes.

La dispersion entre graines décrit la stabilité de l’apprentissage à partition fixe. Elle ne quantifie pas la variabilité entre lieux ou séries de photos. Pour une inférence statistique sur la généralisation, collecte plusieurs groupes indépendants et rééchantillonne les groupes ou réserve un second domaine.

## Comparer la résolution historique à des représentations modernes

Copie `configs/default.json` dans le même dossier pour conserver les chemins relatifs. Crée trois variantes : pixels 6 × 6 avec `image_size: 6`, pixels 16 × 16 et `features: "hog"` avec une taille de 24 ou 32 pixels. Conserve les autres paramètres identiques lors de la première comparaison. Exécute les mêmes modèles avec un nouveau dossier de sortie pour chaque configuration. Vérifie l’identité de `dataset.json`, `splits.json` et `training_indices.json` avant de comparer. Compare sur validation et réserve `finalize` à la configuration figée.

Les pixels conservent la couleur et la position, mais réagissent au cadrage. HOG décrit les orientations locales et perd une partie de la couleur. La PCA réduit la dimension mais peut supprimer des détails discriminants. Pour les arbres, vérifie aussi une représentation sans PCA : leurs seuils univariés peuvent exploiter des pixels utiles directement. Les photos réelles sont nécessaires pour départager ces choix.

## Lire les sorties

- `comparison.csv` donne le classement de validation, le temps, la latence de prédiction en lot et la taille JSON.
- `trials.json` détaille les hyperparamètres, métriques par classe et courbes de coût. Le coût suivi est celui de chaque algorithme. Les valeurs brutes de pertes différentes ne sont pas comparables.
- `figures/` contient les matrices et courbes SVG.
- `test_results.json` associe les erreurs aux chemins relatifs des photos. Relis un échantillon d’erreurs et note les facteurs visibles : cadrage, ombre, occultation, angle, annotation.
- `failures.json` consigne les candidats qui échouent. Un SVM à noyau limité en mémoire doit être comparé sur une sous-cohorte commune, ou déclaré indisponible pour ce benchmark.

Les scores de décision n’ont pas la même échelle entre familles. Une calibration nécessiterait un ensemble dédié ou une validation croisée, une mesure de calibration et un jeu de test figé. L’interface expose la classe prédite pour une exploration locale.
