import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données depuis le fichier Excel
file_path = 'fertilite_senegal_biom.xlsx'
df = pd.read_excel(file_path)

# Nettoyer les noms des colonnes (au cas où il y a des espaces indésirables)
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Variables de sortie (SOC, Biomasse, Fertilité)
output_variables = ['SOC', 'Biomasse', 'Fertilité']

# Calculer la matrice de corrélation entre les variables de sortie et toutes les autres caractéristiques
correlation_with_output = df.corr()[output_variables]

# Tracer la matrice de corrélation entre les variables d'entrée et les variables de sortie
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_with_output, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Matrice de Corrélation - Variables d\'Entrée et Variables de Sortie')
plt.show()

# Afficher la matrice de corrélation entre les variables d'entrée et les variables de sortie
print("Matrice de corrélation entre les variables d'entrée et les variables de sortie:")
print(correlation_with_output)
