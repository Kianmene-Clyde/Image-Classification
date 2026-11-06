# SignLab — Rapport technique et expérimental

Projet annuel IABD ESGI · Classification de panneaux · Clyde Kianmene Takuefou

Le laboratoire compare 23 classifieurs Python sur trois catégories de panneaux et propose quatre variantes C++ accessibles depuis Python. Le protocole fixe les partitions par groupe, ajuste le prétraitement sur l’entraînement et réserve un test à la fin de l’étude. La validation disponible porte sur des données de contrôle synthétiques.

Le fichier `verification-synthetique.json` consigne les mesures de vérification. Le rapport historique `noteRapport.docx` et le syllabus `SyllabusDuProjet.pdf` donnent le contexte d’origine.

## 01 — Problème et périmètre

À partir d’une photo cadrée d’un panneau, prédire l’une des trois classes STOP, sens interdit et limitation à 50 km/h. L’unité d’entrée est une image. L’unité indépendante à considérer pour l’évaluation est le groupe de scène ou le panneau physique quand on connaît cette information.

Le problème relève de la classification supervisée. Une photo contenant plusieurs panneaux, aucun panneau ou un panneau hors des trois classes appelle un autre protocole. Le modèle renvoie toujours l’une des classes connues. Le service local s’adresse à l’expérimentation et à la démonstration pédagogique.

La démarche teste trois idées. Les modèles linéaires donnent une référence peu coûteuse. Les frontières non linéaires peuvent capturer les interactions entre détails du panneau. La représentation des images peut influer davantage sur le score que le choix du classifieur. Les mesures synthétiques vérifient le fonctionnement du protocole. Les photos réelles sont nécessaires pour éprouver ces idées sur l’usage visé.

## 02 — Audit des sources et statut des preuves

Le script historique dépendait d’un dossier local et s’arrêtait avant certains modèles. Le module nommé SVM optimisait une entropie croisée softmax. Dans le MLP, la perte et la rétropropagation ne suivaient pas le même objectif. Les cibles tantôt scalaires, tantôt vectorielles rendaient des comparaisons de classes incorrectes.

L’archive contient un rapport de notes, des poids JSON, des journaux, des scripts Python, deux sources C++ et un syllabus. Les anciens artefacts ne suffisent pas à recréer les partitions ni les transformations. Les scores qui y figurent restent des résultats historiques non reproduits. Leur comparaison numérique avec le nouveau protocole requerrait les mêmes images, les mêmes partitions et un exécutable de référence.

Le parcours opérationnel repose sur le paquet `signlab`. Les sources initiales sont conservées à titre de référence. Le correctif introduit une interface commune, des tests et un mécanisme de sauvegarde qui embarque la transformation des images.

## 03 — Données, provenance et qualité

Les trois répertoires sont prêts pour les photos réelles. Le dépôt d’une image dans un répertoire fournit son étiquette. Le CSV facultatif associe chaque chemin relatif à un groupe de scène et documente la source et la licence. La diversité des origines et la qualité des étiquettes doivent être examinées par classe.

Le chargeur décode les images et standardise l’orientation et les canaux. Un identifiant SHA-256 repère les contenus identiques. Une heuristique combine une empreinte de différences en niveaux de gris et la ressemblance colorimétrique d’une miniature pour rapprocher des quasi-doublons. Des vues éloignées d’un même objet peuvent échapper à la détection automatique.

Un même contenu étiqueté avec deux classes constitue un conflit et interrompt le chargement. Une image illisible est signalée et retirée. En relisant ces incidents, le responsable du dataset peut corriger les fichiers ou les groupes et relancer une expérience avec une nouvelle empreinte.

[figure:flow]

## 04 — Partitions et budget commun

La partition cherche une distribution des classes proche de la cible 60 % entraînement, 20 % validation et 20 % test. Elle assigne ensemble tous les membres d’un groupe. Il faut au minimum trois groupes indépendants par classe. Une structure de groupes très contrainte peut empêcher la présence de toutes les classes dans les trois ensembles.

La validation sélectionne les hyperparamètres. Le test évalue les modèles figés. Toute comparaison de résolutions reprend le même inventaire, la même graine de séparation et les mêmes groupes. Les indices enregistrés permettent de vérifier cette identité.

Le réglage standard limite l’entraînement à 2 500 observations afin de conserver les deux SVM à noyau dans la comparaison. La sélection par groupes s’applique à tous les modèles, avec couverture de chaque classe. Les partitions de validation et test restent entières. Dans une étude séparée, un budget plus élevé peut mesurer le gain des modèles moins gourmands, avec les mêmes règles de sélection.

## 05 — Représentations et normalisation

Une image RVB de taille s × s produit 3s² variables. Le cas 6 × 6 reprend la résolution du script historique. À cette taille, la forme globale et les couleurs restent visibles, mais le texte peut disparaître. Le réglage standard utilise 16 × 16. HOG divise l’image en cellules et accumule les orientations des gradients, puis normalise chaque histogramme.

La PCA calcule une base de projection à partir du train centré. Elle réduit le nombre de dimensions, accélère certains modèles et peut perdre des détails de petite variance. Une comparaison sans PCA mesure ce compromis. Les arbres peuvent être plus faciles à expliquer sur des pixels directs.

La standardisation aide les distances et les gradients. L’échelle min–max fournit des variables positives pour les variantes discrètes de Bayes. Les modèles à seuils n’exigent pas cette étape. Les paramètres de transformation sont conservés dans l’artefact et réutilisés pour validation, test et inférence.

## 06 — Référence, voisinage et Bayes

La référence majoritaire estime les fréquences de classe et prédit toujours la plus fréquente. Son rappel est nul pour les classes jamais choisies. Son macro-F1 met en évidence ce comportement, même si l’accuracy devient élevée dans un jeu déséquilibré.

Le centre de classe réduit chaque classe à une moyenne. KNN conserve les observations du train et vote avec les plus proches. Une petite valeur de k préserve les détails locaux et réagit au bruit. Une valeur plus grande stabilise le vote, au risque d’effacer une classe rare. Les distances dépendent fortement de l’échelle et du nombre de variables.

Bayes naïf estime un modèle par variable conditionnellement à la classe. La variante gaussienne apprend moyenne et variance. Bernoulli modélise une présence après seuil. Multinomial traite les valeurs comme fréquences positives. Complement estime les statistiques des observations extérieures à chaque classe. Les corrélations fortes entre pixels limitent l’interprétation probabiliste.

## 07 — Modèles linéaires et discriminants

La régression logistique apprend des scores affines s(x) = xW + b. Softmax transforme les scores en poids normalisés. L’entropie croisée pénalise une faible valeur attribuée à la vraie classe. La régularisation L2 réduit la norme des poids. Le perceptron ne met à jour les poids qu’en cas d’erreur. Ridge approxime les cibles one-hot par moindres carrés avec une pénalité quadratique.

Le SVM linéaire Crammer-Singer compare le score de la vraie classe avec celui du concurrent ayant le score le plus élevé. Il pénalise une marge inférieure à un et applique une pénalité L2. Les observations bien séparées n’ajoutent plus de terme de marge. Le test des gradients évite les points où plusieurs concurrents sont à égalité.

LDA et QDA représentent chaque classe par une loi gaussienne. Une covariance partagée donne des frontières linéaires. Une covariance propre à chaque classe produit des frontières quadratiques. Le retrait vers une matrice sphérique réduit l’instabilité quand le nombre de variables approche le nombre d’exemples.

## 08 — SVM à noyau

Le noyau gaussien représente une similarité locale. Une faible valeur de gamma lisse les frontières. Une grande valeur rapproche chaque décision des observations mémorisées. Le noyau polynomial exploite des interactions jusqu’au degré choisi. Une mise à l’échelle cohérente évite l’explosion des termes de haut degré.

Le solveur SMO ajuste des paires de coefficients du dual. Il conserve la contrainte d’égalité et la boîte imposée par C. Plusieurs classifieurs binaires, chacun opposant une classe aux autres, produisent les scores multiclasse. Les diagnostics enregistrent le résidu de l’égalité duale et la violation maximale des conditions KKT.

La matrice du noyau occupe un espace quadratique en nombre d’images d’entraînement. La limite de 2 500 correspond à environ 50 Mo pour une seule matrice en double précision, avant les structures auxiliaires. Le solveur pédagogique limite ses passes. Une violation KKT élevée indique que le budget ou le réglage doit être revu sur validation.

## 09 — Arbres et agrégation

CART cherche une variable et un seuil réduisant l’impureté des sous-ensembles. Ici, l’algorithme considère des seuils par quantiles quand la variable prend beaucoup de valeurs. Cela borne la recherche. Limiter la profondeur et augmenter le nombre minimum d’observations par feuille réduisent la capacité d’un arbre à mémoriser de petits groupes.

Le bagging agrège plusieurs arbres entraînés sur des tirages bootstrap. La forêt aléatoire ajoute un sous-ensemble de variables candidates à chaque nœud. Extra Trees tire également des seuils aléatoires et utilise ici tout le train par arbre. La diversification vise à diminuer la corrélation entre erreurs des arbres.

Une comparaison utile conserve le nombre d’arbres, la profondeur et la représentation. Les différences de coût et de rappel par classe importent au-delà du score moyen. Les seuils sur les pixels bruts sont plus directement interprétables que des seuils sur des composantes PCA.

## 10 — Boosting

AdaBoost SAMME entraîne des arbres faibles avec des poids par observation. Une erreur augmente le poids du point. La contribution d’un arbre est fonction de son erreur pondérée et du nombre de classes. Si le classifieur faible ne dépasse pas le seuil requis, l’algorithme s’arrête.

Le gradient boosting construit des scores additifs. À chaque étape, les arbres de régression approximent la différence entre la cible one-hot et le softmax courant. Le taux d’apprentissage règle la contribution de chaque étape. Une petite contribution exige davantage d’arbres et peut améliorer la généralisation.

Les deux méthodes construisent leurs modèles séquentiellement. Elles réagissent différemment au bruit des étiquettes et aux observations atypiques. Compare les courbes train et validation en contrôlant la profondeur et le budget. L’analyse des erreurs doit vérifier si les points difficiles représentent de vrais cas utiles ou une annotation à corriger.

## 11 — Réseaux MLP et RBF

Le MLP projette les variables dans une couche cachée tanh puis dans des scores par classe. La rétropropagation différencie l’entropie croisée softmax. La dérivée de tanh s’applique à la couche cachée. Les mini-lots permettent des mises à jour fréquentes et la graine fixe l’initialisation et l’ordre des lots.

Le réseau RBF calcule une base de similarités aux centres obtenus par k-means++. Le paramètre gamma définit la largeur des réponses radiales. La sortie softmax combine ces réponses. Le réseau et le SVM RBF partagent une forme gaussienne, mais le réseau choisit ses centres par regroupement et le SVM apprend les vecteurs de support à partir de la marge.

Des variables bien mises à l’échelle améliorent l’optimisation du MLP et le regroupement du RBF. Une étude de largeur ou de nombre de centres sur validation aide à vérifier le compromis capacité et coût. Les gains mesurés sur un jeu synthétique simple ont une portée limitée.

## 12 — Justesse et cas d’école

Les vérifications numériques comparent gradients analytiques et différences finies pour la régression logistique, la hinge multiclasse et le MLP. Les cas de contrôle incluent des scores extrêmes, des variables constantes, des entrées invalides, des centres répétés et des ensembles de classes simples. Le SVM à noyau est contrôlé sur ses contraintes du dual.

XOR vérifie une différence de capacité. Les deux classes occupent les diagonales opposées d’un carré. Avec ses deux variables initiales, un modèle linéaire ne peut pas les séparer par une seule droite. Un noyau gaussien ou un MLP peut produire une frontière courbe. Le notebook donne les prédictions et l’accuracy d’entraînement sur ces quatre points.

Le test de sauvegarde compare les classes prédites et les scores de décision avant et après rechargement. Le parcours HTTP vérifie le catalogue, l’inférence et le refus des images invalides.

[figure:xor]

## 13 — Benchmark synthétique des 23 modèles

Les figures de contrôle sont dessinées par le générateur local. Les trois classes partagent un fond bruité et diffèrent fortement par leur forme ou leur couleur. La génération est déterministe et les partitions restent identiques entre modèles. Le tableau présente les essais techniques et le score de validation moyen sur les graines déclarées.

[figure:benchmark]

Un score élevé est attendu sur ce problème simplifié. Des résultats proches sur un petit ensemble ne suffisent pas à établir un classement stable. Le choix final sur photos se fondera aussi sur le coût, la facilité de diagnostic, les erreurs de chaque classe et la diversité des sources.

## 14 — Taille d’entraînement et convergence

La courbe de taille utilise des sous-ensembles emboîtés de l’entraînement. La validation reste fixe. Les points correspondent à environ 25 %, 50 %, 75 % et 100 % des observations d’entraînement. Chaque réglage du MLP est évalué séparément sur les graines configurées.

[figure:learning]

Une performance faible sur train et validation peut indiquer un manque de capacité ou une optimisation insuffisante. Un score train élevé avec un score validation plus bas peut signaler un surapprentissage, un décalage de distribution ou des groupes atypiques. Les courbes de coût permettent de voir si un modèle continue à apprendre lorsque le score par classe plafonne.

Sur une petite fixture et des classes séparées par construction, les courbes peuvent rapidement atteindre un plateau. Les observations ne permettent pas de choisir une largeur optimale pour les photos du projet.

## 15 — Étude de représentation

La comparaison technique emploie les mêmes images, groupes et indices effectifs pour trois représentations. Les candidats ont été limités au premier réglage et à une graine. Les scores proviennent exclusivement de la validation. La valeur à 6 × 6 correspond à une reproduction de la résolution historique sous le nouveau protocole.

[figure:representations]

L’expérience contrôle la comparabilité du pipeline. Sur les photos réelles, augmenter la résolution peut préserver les chiffres et la forme, mais accroître la sensibilité à l’arrière-plan et le coût. HOG exploite les contours et peut perdre une information de couleur utile. Les données doivent décider entre ces compromis.

## 16 — Démonstration et artefacts

Le paquet sépare inventaire des images, représentation, prétraitement, estimateurs, protocole et interface. Le benchmark écrit la configuration, l’environnement, les partitions, les essais et les figures. Chaque modèle JSON versionné contient ses paramètres, l’état du prétraitement, l’ordre des classes et l’empreinte des données.

L’application locale fournit une page de comparaison et une API. Le client envoie une image au modèle choisi. Le serveur valide la taille, décode les pixels et réutilise les transformations sauvegardées. Les scores affichés sont des valeurs propres à chaque modèle. Une calibration demanderait une validation dédiée.

La bibliothèque C++17 expose une interface C pour le modèle linéaire, le MLP, le RBF et le SVM. Python la charge via ctypes. Les sources et la commande de compilation sont livrées. Le compilateur C++17 est absent de l’environnement de vérification utilisé pour ce rapport. Les résultats expérimentaux ci-dessus concernent le parcours Python.

## 17 — Menaces à la validité et travaux sur photos

Des scènes répétées entre les partitions gonflent les scores. Le regroupement par scène et la vérification des doublons réduisent ce risque. Les biais de source, de cadrage et de saison exigent des données diversifiées et un échantillon indépendant pour être mesurés. Pour tester le changement de domaine, réserve par exemple un lieu ou un dispositif de capture complet.

Après l’ajout des images, vérifie les classes et les partitions. Compare les résolutions sur validation et examine les confusions. Fais varier un paramètre à la fois pour attribuer ses effets. Quand le protocole est figé, exécute le test et documente la provenance, la taille des groupes, les erreurs et les cas rares.

Les notes historiques restent dans le dossier de projet. Une comparaison avant et après sur des photos demanderait de reconstruire l’environnement et les transformations utilisés à l’origine. Ce rapport documente la reproductibilité du nouveau parcours et l’état des preuves disponibles.

## 18 — Reproduire l’expérience et références

La procédure démarre avec un environnement Python 3.12 et les versions indiquées dans `requirements-verified.txt`. Crée une fixture avec `python scripts/synthetic_demo.py --output runs/fixture`, puis lance `python -m signlab benchmark --config runs/fixture/config.json --output runs/experience`. Le guide décrit l’audit, le notebook, la finalisation et la compilation native.

Le syllabus du Projet annuel 3 Big Data définit les livrables et les algorithmes à implémenter. Les notes `noteRapport.docx` conservent les expériences initiales. Les sorties de la vérification récente sont réunies dans `docs/verification-synthetique.json`. Les fichiers JSON de l’expérience locale permettent d’auditer les chiffres et les choix en détail.

Pour approfondir les familles de modèles, consulte la documentation officielle scikit-learn sur l’apprentissage supervisé. Pour l’aléatoire reproductible et la liaison native, consulte les références officielles NumPy Generator et Python ctypes. Les liens figurent dans le guide des algorithmes et le README. Le code d’entraînement du catalogue utilise NumPy et ne dépend pas d’estimateurs externes.
