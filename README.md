# Résolution de l'Équation de Poisson 2D par Différences Finies

> **Ce projet a été réalisé dans le cadre du cours d'analyse numérique du Master 1 Génie Informatique à l'École Polytechnique de Yaoundé.**

Ce projet implémente une solution numérique pour l'équation de Poisson 2D sur un domaine carré unitaire $[0,1] \times [0,1]$ en utilisant la méthode des différences finies. Le code permet d'assembler la matrice du système, le second membre, de résoudre le système linéaire résultant, et d'analyser la convergence de la méthode pour différents cas.

## Table des Matières

1.  [Description du Projet](#description-du-projet)
2.  [Structure des Fichiers](#structure-des-fichiers)
3.  [Installation](#installation)
4.  [Utilisation](#utilisation)
5.  [Cas de Test et Résultats](#cas-de-test-et-résultats)
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

## 2. Structure des Fichiers

```
.
├── analysis/
│   ├── convergence_study.py      # Script principal pour l'étude de convergence et la génération des résultats
│   └── results/                  # Dossier contenant les résultats (images, CSV, TXT)
├── src/
│   └── finite_differences_2d.py  # Fonctions d'assemblage de la matrice, du second membre, et création de la grille
├── requirements.txt              # Dépendances Python
├── README.md                     # Ce fichier
```

## 3. Installation

1. **Cloner le dépôt**  
   ```bash
   git clone https://github.com/ThePerformer0/Resolution-EDO-Diff-rence-finies-2D.git
   cd Resolution-EDO-Différence-finies-2D
   ```

2. **Installer les dépendances**  
   Il est recommandé d'utiliser un environnement virtuel :
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## 4. Utilisation

Lance l'étude de convergence et la génération des visualisations avec :
```bash
python analysis/convergence_study.py
```
Les résultats (tableaux de convergence, graphiques, images 3D) seront générés dans le dossier `analysis/results/`.

## 5. Cas de Test et Résultats

Deux cas tests sont inclus pour valider la méthode :

### Cas 1 : $u(x, y) = \sin(\pi x)\sin(\pi y)$
- **Terme source** : $f(x, y) = 2\pi^2 \sin(\pi x)\sin(\pi y)$
- **Conditions aux limites** : $u = 0$ sur le bord
- **Solution exacte** : $u(x, y)$

### Cas 2 : $u(x, y) = x^2 + y^2$
- **Terme source** : $f(x, y) = -4$
- **Conditions aux limites** : $u(x, y) = x^2 + y^2$ sur le bord
- **Solution exacte** : $u(x, y)$

#### Résultats de l'étude de convergence

Un résumé des erreurs $L_\infty$ et des ordres de convergence obtenus :

```
================================================================================
ANALYSE DE CONVERGENCE - MÉTHODE DES DIFFÉRENCES FINIES
================================================================================

Cas: Cas 1: u(x,y) = sin(pi*x)sin(pi*y)
------------------------------------------------------------
N        Erreur L-infini       Ordre de conv. 
------------------------------------------------------------
10        8.2654169662e-03   N/A            
20        2.0587067645e-03   2.0053         
40        5.1420047815e-04   2.0013         
80        1.2852038354e-04   2.0003         
160       3.2128237803e-05   2.0001         
320       8.0319433360e-06   2.0000         
------------------------------------------------------------
Ordre moyen de convergence: 2.0014

Cas: Cas 2: u(x,y) = x^2 + y^2
------------------------------------------------------------
N        Erreur L-infini       Ordre de conv. 
------------------------------------------------------------
10        4.4408920985e-16   N/A            
20        1.1102230246e-15   -1.3219        
40        6.4392935428e-15   -2.5361        
80        1.2878587086e-14   -1.0000        
160       1.3100631691e-14   -0.0247        
320       6.3171690101e-14   -2.2696        
------------------------------------------------------------
Ordre moyen de convergence: -1.4305
```

- **Cas 1** : On observe un ordre de convergence proche de 2, ce qui confirme la précision attendue de la méthode des différences finies d'ordre 2.
- **Cas 2** : L'erreur est de l'ordre de la précision machine (proche de zéro), car la solution exacte est un polynôme du second degré, parfaitement résolue par le schéma.

Des visualisations 3D des solutions numériques, analytiques et des erreurs, ainsi que des graphiques log-log de convergence, sont générées dans `analysis/results/`.

## 6. Dépendances

- numpy
- scipy
- matplotlib
- pandas

Installez-les via :
```bash
pip install -r requirements.txt
```

## 7. Licence

Ce projet est distribué sous licence MIT.

---

**Contact** : Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue ou à me contacter.