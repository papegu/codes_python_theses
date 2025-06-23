import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import os

# Charger les données initiales
file_path = 'fertilite_senegal_biom.xlsx'  # Fichier source
df = pd.read_excel(file_path)

# Nettoyer les noms de colonnes (remplace les espaces par des underscores)
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Définir les colonnes catégorielles
cat_cols = ['Culture_Adaptée', 'Région']

# Définir les colonnes numériques (tout sauf les catégorielles + 'Fertilité')
num_cols = df.columns.difference(cat_cols + ['Fertilité']).tolist() + ['Fertilité']

# Encoder les colonnes catégorielles en codes numériques
df['Culture_Adaptée'] = df['Culture_Adaptée'].astype('category')
df['Culture_Adaptée_Code'] = df['Culture_Adaptée'].cat.codes
culture_mapping = dict(enumerate(df['Culture_Adaptée'].cat.categories))

df['Région'] = df['Région'].astype('category')
df['Région_Code'] = df['Région'].cat.codes
region_mapping = dict(enumerate(df['Région'].cat.categories))

# Cible du nombre total de lignes augmentées
target_rows = 5000

# Création des indices pour l'interpolation
original_indices = np.linspace(0, 1, len(df))
target_indices = np.linspace(0, 1, target_rows)

# Interpolation spline pour les colonnes numériques
new_data = {}
for col in num_cols:
    try:
        cs = CubicSpline(original_indices, df[col])
        new_data[col] = cs(target_indices)
    except Exception as e:
        print(f"Erreur avec la colonne {col} : {e}")

# Interpolation nearest neighbor pour les colonnes catégorielles
interp_culture = interp1d(original_indices, df['Culture_Adaptée_Code'].values,
                          kind='nearest', fill_value="extrapolate")
new_culture_codes = interp_culture(target_indices).astype(int)
new_culture_labels = [culture_mapping[code] for code in new_culture_codes]

interp_region = interp1d(original_indices, df['Région_Code'].values,
                         kind='nearest', fill_value="extrapolate")
new_region_codes = interp_region(target_indices).astype(int)
new_region_labels = [region_mapping[code] for code in new_region_codes]

# Reconstruction du DataFrame final
df_augmented = pd.DataFrame(new_data)
df_augmented['Culture_Adaptée'] = new_culture_labels
df_augmented['Région'] = new_region_labels

# Arrondir l'année si présente
if 'Année' in df_augmented.columns:
    df_augmented['Année'] = df_augmented['Année'].round().astype(int)

# Sauvegarde
output_path = r'D:\dataset12mai.xlsx'
df_augmented.to_excel(output_path, index=False)

print(f"✅ Données augmentées sauvegardées dans : {output_path}")
