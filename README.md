# Résolution de l'Équation de Poisson 2D par Différences Finies

Ce projet implémente une solution numérique pour l'équation de Poisson 2D sur un domaine carré unitaire $[0,1] \times [0,1]$ en utilisant la méthode des différences finies. Le code permet d'assembler la matrice du système, le second membre, de résoudre le système linéaire résultant, et d'analyser la convergence de la méthode pour différents cas.

## Table des Matières

1.  [Description du Projet](#description-du-projet)
2.  [Structure des Fichiers](#structure-des-fichiers)
3.  [Installation](#installation)
4.  [Utilisation](#utilisation)
    * [Exemple Simple](#exemple-simple)
    * [Étude de Convergence](#étude-de-convergence)
5.  [Résultats et Visualisations](#résultats-et-visualisations)
6.  [Dépendances](#dépendances)
7.  [Licence](#licence)

## 1. Description du Projet

L'équation de Poisson 2D est donnée par :
$$
-\Delta u = f(x, y) \quad \text{sur} \quad \Omega = [0,1] \times [0,1]
$$
avec des conditions aux limites de Dirichlet $u(x, y) = g(x, y)$ sur la frontière $\partial\Omega$.

Ce projet discrétise le domaine en utilisant une grille uniforme et approxime le laplacien en utilisant des différences finies centrées d'ordre 2. Cela conduit à un système linéaire d'équations qui est résolu pour obtenir la solution numérique.

Les fonctionnalités clés incluent :
* **Assemblage de la matrice du système (A) :** Construction de la matrice creuse représentant l'opérateur Laplacien discret.
* **Assemblage du vecteur du second membre (B) :** Intégration du terme source $f(x,y)$ et des conditions aux limites $g(x,y)$.
* **Résolution du système linéaire :** Utilisation de solveurs creux pour obtenir la solution numérique $U$.
* **Visualisation 2D et 3D :** Représentation des solutions numériques, analytiques et des erreurs.
* **Étude de convergence :** Analyse de l'ordre de convergence de la méthode en calculant l'erreur $L_\infty$ pour différentes tailles de grille.