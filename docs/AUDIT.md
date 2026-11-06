# Audit et état du projet

| Priorité | Constat dans le dépôt initial | Correction ou état |
| --- | --- | --- |
| P0 | `tests/libraryTest.py` pointe sur un disque personnel. Un `exit(0)` rend plusieurs expériences inaccessibles. | Point d’entrée portable. Les sources initiales sont conservées dans `legacy/`. |
| P0 | `models/python/SVM.py` minimise l’entropie croisée softmax, caractéristique d’une régression logistique. | Deux modèles séparés. Le SVM linéaire utilise une perte de marge de Crammer-Singer. Les variantes noyau utilisent un dual C-SVM. |
| P0 | Le MLP historique mélange one-hot et indices, compare un score ou un indice à un vecteur et applique des dérivées incohérentes. | Contrat de cibles unidimensionnelles. Entropie croisée et gradients vérifiés par différences finies. |
| P0 | Le modèle linéaire historique recalcule moyenne et écart-type à chaque prédiction. Le RBF historique peut calculer une dispersion sur une grappe vide. | Prétraitement appris sur train et conservé. Échelles nulles protégées. Centres RBF et gradients validés sur cas synthétiques. |
| P0 | Le ZIP ne contient pas les images référencées par le rapport historique. | Dossiers vides fournis. Tests techniques séparés. Entraînement sur photos et comparaison historique en attente des données. |
| P1 | Le script historique ne conserve pas de validation indépendante pour le choix des hyperparamètres. | Séparation groupée train, validation et test. Le score de test est produit par une commande distincte après gel des choix. |
| P1 | Des paramètres, états et seuils sont dispersés. Les modèles sauvegardés n’embarquent pas un prétraitement exploitable. | Catalogue, configuration JSON, empreinte des images et état sauvegardé avec chaque modèle. |
| P1 | Le syllabus demande quatre modèles personnels natifs. Deux sources C++ historiques sont présentes sans chaîne de compilation commune. | Quatre classifieurs C++17 avec une interface C, liaison Python et tests conditionnels. |
| P2 | L’API et l’interface sont annoncées dans le README initial. | Une interface locale expose catalogue, comparaison et prédiction. |
| P2 | Les notes présentent des scores historiques avec des protocoles et hyperparamètres variables. | Rapport séparant les résultats historiques, la validation technique et le futur benchmark réel. |

Les sources historiques restent consultables pour retracer les contributions. Elles ne participent pas au parcours SignLab. Les artefacts historiques ne peuvent pas être importés comme nouveaux modèles sans retrouver les prétraitements et l’ordre de classes utilisés à l’époque.

La bibliothèque C++ est distincte des 23 modèles Python du benchmark par défaut. Les deux implémentations partagent la tâche et les partitions, mais certains détails d’optimisation diffèrent, notamment l’apprentissage par lot complet du MLP C++.

## Validation

- Les 23 modèles Python ont été entraînés et sauvegardés sur la même fixture de 72 images. Les 120 essais de validation et l'évaluation finale technique figurent dans [les résultats synthétiques](verification-synthetique.json).
- Quinze tests ont réussi. Le test des quatre modèles natifs a été ignoré faute de compilateur C++17 sur la machine de vérification.
- Les cas synthétiques vérifient notamment la séparation linéaire, XOR, les gradients, les contraintes du dual SVM, la conservation des prédictions à la lecture des artefacts et le parcours HTTP.
- Les réglages du catalogue sont pédagogiques et bornés. Le SMO est simplifié et indique la violation KKT. Les arbres testent des seuils candidats par quantiles pour les variables continues.
- Les résultats historiques sont invérifiables sans les photos, les partitions et l’environnement d’origine. Les résultats synthétiques servent à vérifier l’exécution.
