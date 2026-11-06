"""Catalogue central : constructeurs, variantes et domaine du prétraitement."""

from dataclasses import dataclass
from .models.simple import Majority, NearestCentroid, KNN, NaiveBayes
from .models.linear import Logistic, Perceptron, RidgeClassifier, LinearSVM, Discriminant
from .models.nonlinear import KernelSVM, MLP, RBFNetwork
from .models.trees import DecisionTree, Forest, AdaBoost, GradientBoosting
from .native import NativeClassifier


@dataclass(frozen=True)
class Spec:
    cls: type
    family: str
    description: str
    candidates: tuple
    scale: str = "standard"


CATALOG = {
    "majority": Spec(Majority, "Référence", "Prédit toujours la classe majoritaire.", ({},), "none"),
    "nearest_centroid": Spec(NearestCentroid, "Voisinage", "Compare les distances aux moyennes de classes.", ({},)),
    "knn": Spec(KNN, "Voisinage", "Vote local des voisins ; coût reporté sur l'inférence.", ({"n_neighbors": 3}, {"n_neighbors": 7})),
    "gaussian_nb": Spec(NaiveBayes, "Bayes", "Variables gaussiennes indépendantes conditionnellement à la classe.", ({"kind": "gaussian"},)),
    "bernoulli_nb": Spec(NaiveBayes, "Bayes", "Pixels binarisés au seuil 0,5 après mise à l'échelle.", ({"kind": "bernoulli"},), "minmax"),
    "multinomial_nb": Spec(NaiveBayes, "Bayes", "Variables positives assimilées à des fréquences : hypothèse forte pour des pixels.", ({"kind": "multinomial"},), "minmax"),
    "complement_nb": Spec(NaiveBayes, "Bayes", "Estime les distributions des classes complémentaires.", ({"kind": "complement"},), "minmax"),
    "logistic": Spec(Logistic, "Linéaire", "Softmax et entropie croisée régularisée ; ce n'est pas un SVM.", ({"l2": 0.001}, {"l2": 0.03})),
    "perceptron": Spec(Perceptron, "Linéaire", "Corrige les erreurs de classification par mises à jour additives.", ({"epochs": 40}, {"epochs": 80})),
    "ridge": Spec(RidgeClassifier, "Linéaire", "Moindres carrés régularisés sur cibles one-hot.", ({"alpha": 1.0}, {"alpha": 10.0})),
    "linear_svm": Spec(LinearSVM, "Marge", "Marge multiclasse hinge de Crammer-Singer et pénalité L2.", ({"l2": 0.01}, {"l2": 0.1})),
    "kernel_svm_rbf": Spec(KernelSVM, "Marge", "C-SVM dual SMO simplifié et noyau gaussien, one-versus-rest.", ({"kernel": "rbf", "C": 1.0}, {"kernel": "rbf", "C": 5.0})),
    "kernel_svm_poly": Spec(KernelSVM, "Marge", "C-SVM à noyau polynomial ; sensible à l'échelle et au degré.", ({"kernel": "poly", "degree": 2}, {"kernel": "poly", "degree": 3})),
    "lda": Spec(Discriminant, "Discriminant", "Covariance commune régularisée : frontières linéaires.", ({"quadratic": False, "shrinkage": 0.2}, {"quadratic": False, "shrinkage": 0.8})),
    "qda": Spec(Discriminant, "Discriminant", "Covariances par classe régularisées : frontières quadratiques.", ({"quadratic": True, "shrinkage": 0.2}, {"quadratic": True, "shrinkage": 0.8})),
    "decision_tree": Spec(DecisionTree, "Arbres", "CART Gini, recherche de seuils par quantiles pour borner le coût.", ({"max_depth": 4}, {"max_depth": 7}), "none"),
    "bagging": Spec(Forest, "Ensembles", "Arbres sur échantillons bootstrap, toutes les variables candidates.", ({"kind": "bagging", "n_estimators": 10}, {"kind": "bagging", "n_estimators": 20}), "none"),
    "random_forest": Spec(Forest, "Ensembles", "Bootstrap et sous-ensemble aléatoire de variables à chaque nœud.", ({"kind": "random_forest", "max_depth": 4}, {"kind": "random_forest", "max_depth": 7}), "none"),
    "extra_trees": Spec(Forest, "Ensembles", "Variables et seuils aléatoires, sans bootstrap.", ({"kind": "extra_trees", "max_depth": 6}, {"kind": "extra_trees", "max_depth": 10}), "none"),
    "adaboost": Spec(AdaBoost, "Boosting", "SAMME : augmente le poids des observations mal classées.", ({"n_estimators": 20}, {"n_estimators": 40}), "none"),
    "gradient_boosting": Spec(GradientBoosting, "Boosting", "Arbres de régression ajustés aux gradients de l'entropie croisée.", ({"n_estimators": 20}, {"n_estimators": 40}), "none"),
    "mlp": Spec(MLP, "Réseaux", "Une couche cachée tanh, softmax, mini-batch SGD et L2.", ({"hidden_units": 24}, {"hidden_units": 48})),
    "rbf_network": Spec(RBFNetwork, "Réseaux", "Centres k-means++, bases gaussiennes et sortie softmax.", ({"n_centers": 12}, {"n_centers": 32})),
}


def build_model(name, params=None, seed=42):
    if name not in CATALOG:
        raise ValueError(f"Modèle inconnu : {name}. Utilisez 'catalog'.")
    spec = CATALOG[name]
    kwargs = dict(params or spec.candidates[0])
    import inspect
    if "random_state" in inspect.signature(spec.cls).parameters:
        kwargs["random_state"] = int(seed)
    return spec.cls(**kwargs)


for name, kind in (("linear", 0), ("svm", 1), ("mlp", 2), ("rbf", 3)):
    CATALOG[f"native_{name}"] = Spec(NativeClassifier, "C++ personnel", "Bibliothèque C++17 via ctypes ; compilation préalable nécessaire.", ({"kind": kind},))

SERIALIZABLE_CLASSES = {spec.cls.__name__: spec.cls for spec in CATALOG.values()}
