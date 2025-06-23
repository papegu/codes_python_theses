import pandas as pd
import numpy as np

def augmenter_lignes(fichier_excel, cible_lignes=1000):
    # Charger le fichier Excel dans un DataFrame
    df = pd.read_excel(fichier_excel)

    # Calculer le nombre de lignes à ajouter pour atteindre 1000
    lignes_initiales = len(df)
    
    if lignes_initiales >= cible_lignes:
        print("Le nombre de lignes est déjà supérieur ou égal à 1000.")
        return df
    
    # Calculer le nombre de répétitions nécessaires pour atteindre le nombre de lignes souhaité
    repetitions = (cible_lignes // lignes_initiales) + 1  # Répéter les données plus d'une fois si nécessaire

    # Répéter les lignes du DataFrame pour atteindre au moins 1000 lignes
    df_augmenté = pd.concat([df] * repetitions, ignore_index=True)

    # Si le DataFrame est plus grand que 1000 lignes, on le découpe pour obtenir exactement 1000 lignes
    df_augmenté = df_augmenté.head(cible_lignes)

    # Sauvegarder le DataFrame augmenté dans un nouveau fichier Excel
    df_augmenté.to_excel("fichier_augmenté.xlsx", index=False)

    print(f"Le fichier a été augmenté pour atteindre {len(df_augmenté)} lignes.")
    return df_augmenté

# Exemple d'utilisation
file_path = 'proprietes_sol_estimees_senegal1.xlsx'  # Remplace par le chemin vers ton fichier Excel
df_augmenté = augmenter_lignes(file_path, cible_lignes=1000)
