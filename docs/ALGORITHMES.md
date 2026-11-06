# Choisir et comprendre un classifieur

Les 23 entrées couvrent les principales familles classiques adaptées à une classification supervisée multiclasse. Chaque variante est entraînable depuis la même commande. Les régressions de cibles continues, le clustering et la détection d’anomalies résolvent d’autres tâches. K-means apparaît ici comme étape interne du réseau RBF et PCA comme prétraitement.

| Nom CLI | Idée et hyperparamètre à manipuler | Limite à observer |
| --- | --- | --- |
| `majority` | Classe majoritaire du train, sans variables. | L’accuracy peut être trompeuse si une classe domine. |
| `nearest_centroid` | Un centre par classe, distance euclidienne. | Deux modes très éloignés au sein d’une classe sont mal représentés. |
| `knn` | Vote des k voisins avec pondération par distance. Fais varier k. | Stocke le train, prédiction coûteuse, sensible à l’échelle et aux dimensions inutiles. |
| `gaussian_nb` | Moyennes et variances par classe et par variable. Ajuste le plancher de variance. | Les pixels voisins sont corrélés. Les scores deviennent parfois excessivement affirmés. |
| `bernoulli_nb` | Seuil 0,5, puis présence ou absence par variable. Ajuste le lissage. | La binarisation perd une partie de la couleur et de la texture. |
| `multinomial_nb` | Log de fréquences relatives avec lissage alpha. | L’hypothèse de comptage est approximative sur des intensités et des composantes mises à l’échelle. |
| `complement_nb` | Apprend la distribution en dehors de chaque classe. Compare sur un jeu déséquilibré. | Deux images de proportions voisines peuvent produire des décisions semblables malgré des intensités différentes. |
| `logistic` | Scores linéaires, softmax, entropie croisée et pénalité L2. Fais varier L2. | Une frontière linéaire ne résout pas XOR. |
| `perceptron` | Corrige le vecteur de la classe prédite et celui de la vraie classe après une erreur. | L’ordre des exemples compte. Des classes non séparables peuvent empêcher la convergence. |
| `ridge` | Ajuste des cibles one-hot par moindres carrés régularisés. Fais varier alpha. | Les scores sont des sorties de régression. |
| `linear_svm` | Minimise la marge hinge du concurrent le plus dangereux, avec L2. | Les exemples bien au-delà de la marge ne contribuent plus au gradient de la perte. |
| `kernel_svm_rbf` | Produit des similarités locales exp(−γ‖x−z‖²). Ajuste C et gamma. | La matrice noyau est quadratique et une valeur gamma trop élevée isole les observations. |
| `kernel_svm_poly` | Apprend avec des interactions via (γ x·z + coef0)^degré. | Le degré amplifie les différences d’échelle et peut dégrader l’optimisation. |
| `lda` | Gaussiennes partageant une covariance. Ajuste le retrait vers une covariance sphérique. | La frontière reste linéaire. Une covariance commune peut masquer des différences. |
| `qda` | Une covariance par classe. Compare son retrait à celui de LDA. | Estimation plus fragile quand les classes ont peu de données. |
| `decision_tree` | Divisions par seuil réduisant l’impureté de Gini. Ajuste la profondeur. | Les partitions sont alignées sur les axes et la variance augmente avec la profondeur. |
| `bagging` | Plusieurs arbres sur des échantillons bootstrap. Ajuste leur nombre. | Des arbres très corrélés donnent moins de gain au vote. |
| `random_forest` | Bootstrap et tirage de variables à chaque nœud. | L’échantillonnage de variables peut manquer un détail rare. |
| `extra_trees` | Variables et seuils tirés aléatoirement. | L’aléa peut augmenter le biais, notamment avec trop peu d’arbres. |
| `adaboost` | SAMME augmente le poids des erreurs. Ajuste profondeur et nombre de tours. | Les erreurs d’étiquetage deviennent progressivement influentes. |
| `gradient_boosting` | Arbres ajustés aux résidus one-hot moins softmax. Ajuste taux et nombre d’étapes. | Entraînement séquentiel, sensible à la combinaison profondeur–taux. |
| `mlp` | Couche cachée tanh, puis sortie softmax. Compare largeur, taux et nombre d’époques. | Le résultat dépend de l’initialisation, de l’échelle et de la régularisation. |
| `rbf_network` | Centres k-means++, similarité radiale aux centres, sortie softmax. | Des centres non supervisés peuvent représenter le fond plus que le panneau. |

## Expériences pour comprendre les différences

1. Exécute le notebook sur XOR. Observe la différence entre la frontière linéaire et le noyau RBF. Le MLP peut aussi modéliser XOR avec une couche cachée. Entraîne plusieurs graines pour distinguer capacité et difficulté d’optimisation.
2. Compare un arbre peu profond à un arbre plus profond dans `trials.json`. Un écart train–validation croissant suggère un problème de généralisation. Vérifie la distribution des erreurs avant de conclure.
3. Compare le SVM RBF au réseau RBF. Le premier sélectionne des vecteurs de support au moyen d’une marge. Le second construit des centres non supervisés et ajuste une sortie softmax.
4. Compare KNN et centre de classe. Une image rare proche d’un voisin pertinent peut être bien classée par KNN, même si son centre de classe est éloigné.
5. Compare LDA et QDA sur des classes de tailles différentes. Vérifie si la covariance estimée à partir du petit groupe aide ou nuit sur la validation.
6. Compare forêt, bagging et Extra Trees avec la même représentation. Les sources de diversité diffèrent : observations, variables et seuils.
7. Compare la régression logistique et le SVM linéaire sur une même partition. Sur une observation aisée, la perte logistique décroît progressivement. La hinge devient nulle dès que la marge requise est atteinte.

Les paramètres testés sont dans `signlab/registry.py`. Le guide officiel [des familles supervisées de scikit-learn](https://scikit-learn.org/stable/supervised_learning.html) permet d’approfondir ces méthodes. Les modèles du catalogue sont des implémentations pédagogiques indépendantes. Leurs graines passent par les [générateurs locaux NumPy](https://numpy.org/doc/stable/reference/random/generator.html).
