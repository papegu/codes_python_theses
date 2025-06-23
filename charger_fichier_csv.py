import pandas as pd

# Charger les données du fichier CSV
df_csv = pd.read_csv('crop_recommendation.csv')

# Afficher les premières lignes pour inspecter la structure
print(df_csv.head())
