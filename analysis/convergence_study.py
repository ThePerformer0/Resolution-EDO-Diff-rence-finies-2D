import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Pour les tracés 3D
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import math
import os
import sys

# --- Configuration pour l'importation de modules locaux ---
# Assurez-vous que le répertoire 'src' est dans le PYTHONPATH
# En supposant que finite_differences_2d.py est dans un dossier 'src' au même niveau que ce script.
current_dir = os.path.dirname(os.path.abspath(__file__))
# Ajoutez le répertoire parent (où 'src' devrait être) au chemin système
# Ou si 'src' est au même niveau que 'convergence_study.py', ajoutez simplement current_dir
# Pour cette structure (convergence_study.py et src/finite_differences_2d.py au même niveau):
# sys.path.insert(0, os.path.join(current_dir, 'src'))
# Correction: Si src est un sous-dossier, l'import doit être `from src.module_name`
# Si src est au même niveau et vous voulez `import module_name`, alors il faut ajouter le parent.
# Étant donné que le problème persiste, la meilleure approche est d'ajouter le répertoire parent
# afin que 'src' puisse être importé comme un package.

# Obtenir le chemin du répertoire parent (où 'src' se trouve si votre structure est main_folder/src/...)
parent_dir = os.path.dirname(current_dir)
# Ajouter le répertoire parent au chemin de recherche des modules
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# Maintenant, nous pouvons importer depuis src
try:
    from src.finite_differences_2d import assemble_matrix, assemble_rhs, create_grid
    print("Module 'finite_differences_2d' importé avec succès.")
except ImportError as e:
    print(f"Erreur d'importation du module: {e}")
    print("Veuillez vous assurer que le fichier 'finite_differences_2d.py' se trouve dans un dossier 'src' au même niveau que ce script.")
    print(f"Le chemin actuel de sys.path est: {sys.path}")
    sys.exit(1) # Quitter si l'importation échoue


# --- Définir les fonctions du problème pour le Cas 1: u(x,y) = sin(pi*x)sin(pi*y) ---
def f_func_cas1(x, y):
    """Terme source pour le Cas 1: u(x,y) = sin(pi*x)sin(pi*y)."""
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

def g_func_cas1(x, y):
    """Conditions aux limites de Dirichlet pour le Cas 1 (u=0 sur la frontière)."""
    return 0.0

def u_exact_func_cas1(x, y):
    """Solution analytique pour le Cas 1."""
    return np.sin(np.pi * x) * np.sin(np.pi * y)

# --- Définir les fonctions du problème pour le Cas 2: u(x,y) = x^2 + y^2 ---
def f_func_cas2(x, y):
    """Terme source pour le Cas 2: u(x,y) = x^2+y^2 => -Delta U = -4."""
    return -4.0

def g_func_cas2(x, y):
    """Conditions aux limites de Dirichlet pour le Cas 2: u(x,y) = x^2+y^2 sur la frontière."""
    return x**2 + y**2

def u_exact_func_cas2(x, y):
    """Solution analytique pour le Cas 2."""
    return x**2 + y**2

# --- Fonction utilitaire pour formater le tableau de convergence TXT ---
def _format_convergence_table_txt(case_name, N_values, l_inf_errors, orders_of_conv, avg_order):
    """Formate les données de convergence dans un tableau texte pour le rapport."""
    table_str = f"Cas: {case_name}\n"
    table_str += "------------------------------------------------------------\n"
    table_str += "N        Erreur L-infini       Ordre de conv. \n"
    table_str += "------------------------------------------------------------\n"

    for i in range(len(N_values)):
        N_val = N_values[i]
        error_val = l_inf_errors[i]
        order_val = orders_of_conv[i]

        order_str = f"{order_val:.4f}" if order_val is not None and not np.isnan(order_val) else "N/A"
        table_str += f"{N_val:<9} {error_val:.10e}   {order_str:<15}\n"

    table_str += "------------------------------------------------------------\n"
    table_str += f"Ordre moyen de convergence: {avg_order:.4f}\n"
    return table_str

# --- Fonction pour générer et sauvegarder les tracés de comparaison (3D) ---
def generate_comparison_plots_3d(N, f_func, g_func, u_exact_func, case_title, filename_prefix, output_dir):
    """
    Génère et enregistre les tracés 3D de la solution numérique, analytique et de l'erreur absolue.
    """
    
    A_sparse = assemble_matrix(N)
    A_csr = A_sparse.tocsr()
    B = assemble_rhs(f_func, g_func, N)
    U_numerical_flat = spsolve(A_csr, B)

    num_interior_points_1d = N - 1
    U_numerical_reshaped = U_numerical_flat.reshape(num_interior_points_1d, num_interior_points_1d)
    U_full_grid = np.zeros((N + 1, N + 1))
    x_grid, y_grid = create_grid(N)
    U_full_grid[1:N, 1:N] = U_numerical_reshaped
    
    # Appliquer les conditions aux limites à la grille complète pour les plots 3D
    # Ces boucles sont essentielles pour que les bords du plot 3D soient corrects.
    for idx in range(N + 1):
        U_full_grid[idx, 0] = g_func(x_grid[idx], y_grid[0])
        U_full_grid[idx, N] = g_func(x_grid[idx], y_grid[N])
        U_full_grid[0, idx] = g_func(x_grid[0], y_grid[idx])
        U_full_grid[N, idx] = g_func(x_grid[N], y_grid[idx])

    X_full, Y_full = np.meshgrid(x_grid, y_grid)
    U_exact_full_grid = u_exact_func(X_full, Y_full)
    error_abs = np.abs(U_full_grid - U_exact_full_grid)

    # Plotting 3D
    fig = plt.figure(figsize=(24, 7)) # Plus large pour 3 subplots 3D

    # Plot Numérique 3D
    ax0 = fig.add_subplot(1, 3, 1, projection='3d')
    ax0.plot_surface(X_full, Y_full, U_full_grid, cmap='viridis', edgecolor='none')
    ax0.set_title(f'Solution Numérique (N={N})\nCas: {case_title}')
    ax0.set_xlabel('x'); ax0.set_ylabel('y'); ax0.set_zlabel('U Numérique')

    # Plot Analytique 3D
    ax1 = fig.add_subplot(1, 3, 2, projection='3d')
    ax1.plot_surface(X_full, Y_full, U_exact_full_grid, cmap='viridis', edgecolor='none')
    ax1.set_title(f'Solution Analytique (N={N})\nCas: {case_title}')
    ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('U Analytique')

    # Plot Erreur Absolue 3D
    ax2 = fig.add_subplot(1, 3, 3, projection='3d')
    ax2.plot_surface(X_full, Y_full, error_abs, cmap='magma', edgecolor='none') # Utilisation de 'magma' pour l'erreur
    ax2.set_title(f'Erreur Absolue (N={N})\nCas: {case_title}')
    ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('Erreur Absolue')

    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f'{filename_prefix}_N{N}_3D_comparison.png')
    plt.savefig(plot_filename)
    plt.close(fig) # Fermer la figure pour libérer de la mémoire
    print(f"  Visualisation 3D pour N={N} enregistrée sous {plot_filename}")


# --- Main convergence study block ---
if __name__ == "__main__":
    print("Démarrage de l'étude de convergence pour l'équation de Poisson 2D...")

    N_values = [10, 20, 40, 80, 160, 320] # N intervalles le long de chaque côté
    all_convergence_data = [] # Pour stocker les données pour le CSV

    # Créer le répertoire pour les résultats s'il n'existe pas
    output_dir = 'analysis/results'
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Définir les cas de test ---
    test_cases = [
        {
            "name": "Cas 1: u(x,y) = sin(pi*x)sin(pi*y)",
            "f_func": f_func_cas1,
            "g_func": g_func_cas1,
            "u_exact_func": u_exact_func_cas1,
            "file_prefix": "cas1_sin_pi_x_sin_pi_y"
        },
        {
            "name": "Cas 2: u(x,y) = x^2 + y^2",
            "f_func": f_func_cas2,
            "g_func": g_func_cas2,
            "u_exact_func": u_exact_func_cas2,
            "file_prefix": "cas2_x_squared_plus_y_squared"
        }
    ]

    txt_output_sections = [] # Pour le fichier TXT

    for case_idx, case in enumerate(test_cases):
        print(f"\n--- Exécution du {case['name']} ---")
        l_inf_errors = []
        orders_of_conv = [None] * len(N_values)
        
        for i, N in enumerate(N_values):
            h = 1.0 / N

            # Assembler et résoudre
            A_sparse = assemble_matrix(N)
            A_csr = A_sparse.tocsr()
            B = assemble_rhs(case["f_func"], case["g_func"], N)
            U_numerical_flat = spsolve(A_csr, B)

            # Reconstruire la solution de la grille complète
            num_interior_points_1d = N - 1
            U_numerical_reshaped = U_numerical_flat.reshape(num_interior_points_1d, num_interior_points_1d)
            U_full_grid = np.zeros((N + 1, N + 1))
            x_grid, y_grid = create_grid(N)
            U_full_grid[1:N, 1:N] = U_numerical_reshaped
            
            # Appliquer les conditions aux limites à la grille complète pour le calcul de l'erreur
            for idx in range(N + 1):
                U_full_grid[idx, 0] = case["g_func"](x_grid[idx], y_grid[0])
                U_full_grid[idx, N] = case["g_func"](x_grid[idx], y_grid[N])
                U_full_grid[0, idx] = case["g_func"](x_grid[0], y_grid[idx])
                U_full_grid[N, idx] = case["g_func"](x_grid[N], y_grid[idx])

            # Calculer la solution analytique
            X_full, Y_full = np.meshgrid(x_grid, y_grid)
            U_exact_full_grid = case["u_exact_func"](X_full, Y_full)

            # Calculer l'erreur L-infini
            error_abs = np.abs(U_full_grid - U_exact_full_grid)
            l_inf_error = np.max(error_abs)
            l_inf_errors.append(l_inf_error)

            # Calculer l'ordre de convergence
            order = None
            if i > 0:
                if l_inf_errors[i-1] > 0 and l_inf_error > 0:
                    order = math.log2(l_inf_errors[i-1] / l_inf_error)
                else:
                    order = np.nan # Utiliser NaN pour les calculs pandas
            orders_of_conv[i] = order
            
            print(f"  N={N}: Erreur L-infini={l_inf_error:.6e}")

            # Générer les images de comparaison 3D pour chaque N
            print(f"  Génération des images de comparaison 3D pour N={N}...")
            generate_comparison_plots_3d(N, case["f_func"], case["g_func"], case["u_exact_func"], 
                                         case["name"], case["file_prefix"], output_dir)
        
        # Calculer l'ordre moyen de convergence pour le cas actuel
        valid_orders = [o for o in orders_of_conv if o is not None and not np.isnan(o)]
        avg_order = np.mean(valid_orders) if valid_orders else 0.0

        # Ajouter les données de convergence à la liste pour le CSV
        for i in range(len(N_values)):
            all_convergence_data.append({
                "Cas de Test": case["name"],
                "N (Points par côté)": N_values[i],
                "h (Taille de Pas)": 1.0 / N_values[i],
                "Erreur L-infini": l_inf_errors[i],
                "Ordre de Convergence": orders_of_conv[i] if orders_of_conv[i] is not None else np.nan
            })
        
        # Préparer la section pour le fichier TXT
        txt_output_sections.append(_format_convergence_table_txt(
            case["name"], N_values, l_inf_errors, orders_of_conv, avg_order
        ))
        
        # --- Tracer le taux de convergence (graphique log-log) pour chaque cas séparément ---
        plt.figure(figsize=(9, 7)) # Figure pour un seul cas
        plt.loglog([1.0/N for N in N_values], l_inf_errors, 'o-', label=f'Erreur L-infini Numérique')
        
        plt.title(f'Analyse de Convergence - {case["name"]}')
        plt.xlabel('Pas de discrétisation h (échelle log)')
        plt.ylabel('Erreur L-infini (échelle log)')
        plt.grid(True, which="both", ls="-")

        # Ajouter une ligne de référence pour la convergence d'ordre 2 (pente = 2)
        if len(N_values) > 1 and l_inf_errors[0] > 0:
            h_ref = np.array([1.0/N_values[0], 1.0/N_values[-1]])
            error_ref_slope2 = l_inf_errors[0] * (h_ref / (1.0/N_values[0]))**2
            plt.loglog(h_ref, error_ref_slope2, 'k--', label='Pente de référence 2 (ordre 2)')

        plt.legend()
        plot_convergence_filename = os.path.join(output_dir, f'convergence_plot_{case["file_prefix"]}.png')
        plt.savefig(plot_convergence_filename)
        plt.close(plt.gcf()) # Fermer la figure actuelle
        print(f"  Graphique de convergence (L-infini) pour {case['name']} enregistré sous {plot_convergence_filename}")


    # --- Générer la sortie texte formatée (combinée) ---
    output_text = "================================================================================\n"
    output_text += "ANALYSE DE CONVERGENCE - MÉTHODE DES DIFFÉRENCES FINIES\n"
    output_text += "================================================================================\n\n"
    output_text += "\n\n".join(txt_output_sections)

    # Enregistrer dans un fichier TXT
    txt_filename = os.path.join(output_dir, 'convergence_analysis_summary.txt')
    with open(txt_filename, 'w') as f:
        f.write(output_text)
    print(f"\nLes résultats de l'analyse de convergence ont été enregistrés dans {txt_filename}")

    # --- Enregistrer les données de convergence dans un CSV plus lisible ---
    df_convergence = pd.DataFrame(all_convergence_data)
    # Remplacer NaN pour 'Ordre de Convergence' par une chaîne vide pour une meilleure lisibilité dans le CSV
    df_convergence['Ordre de Convergence'] = df_convergence['Ordre de Convergence'].fillna('N/A')
    
    csv_filename = os.path.join(output_dir, 'convergence_results.csv')
    df_convergence.to_csv(csv_filename, index=False)
    print(f"\nLes données de convergence complètes ont été enregistrées dans {csv_filename}")

    print("\nL'étude de convergence est terminée pour tous les cas.")

    # Afficher le contenu du fichier .txt généré
    with open(txt_filename, 'r') as f:
        print("\n--- Contenu de convergence_analysis_summary.txt ---")
        print(f.read())

    # Afficher les premières lignes du DataFrame pour le CSV
    print("\n--- Aperçu des données de convergence (CSV) ---")
    print(df_convergence.head(10))