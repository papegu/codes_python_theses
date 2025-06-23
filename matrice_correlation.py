import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données depuis le fichier Excel
file_path = 'fertilite_senegal_biom.xlsx'
df = pd.read_excel(file_path)

# Nettoyer les noms des colonnes (au cas où il y a des espaces indésirables)
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Matrice de corrélation entre les propriétés (SOC, Biomasse, Fertilité)
correlation_properties = df[['SOC', 'Biomasse', 'Fertilité']].corr()

# Matrice de corrélation entre les propriétés du sol (pH, Argile, etc.)
correlation_soil_properties = df[['pH', 'Argile', 'Matière_Organique', 'Azote_(N)', 'Phosphore_(P)', 'Potassium_(K)', 'Latitude', 'Longitude']].corr()

# Tracer la matrice de corrélation entre les propriétés
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_properties, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Matrice de Corrélation - SOC, Biomasse, Fertilité')
plt.show()

# Tracer la matrice de corrélation entre les propriétés du sol
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_soil_properties, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Matrice de Corrélation - Propriétés du Sol')
plt.show()

# Afficher la matrice de corrélation entre SOC, Biomasse et Fertilité
print("Matrice de corrélation entre SOC, Biomasse et Fertilité:")
print(correlation_properties)
