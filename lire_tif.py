import rasterio
import pandas as pd
import numpy as np

def tif_to_excel(tif_file, excel_file):
    # Lire le fichier .tif
    with rasterio.open(tif_file) as src:
        # Lire les données de l'image en tant que matrice
        data = src.read(1)  # Lire la première bande (si plusieurs bandes, tu peux changer l'index)
        
        # Créer un DataFrame pandas à partir des données
        df = pd.DataFrame(data)
        
        # Sauvegarder le DataFrame en fichier Excel
        df.to_excel(excel_file, index=False, header=False)

# Exemple d'utilisation
tif_file = "out.tif"  # Remplace par le chemin vers ton fichier .tif
excel_file = "output_file.xlsx"     # Le fichier Excel de sortie

tif_to_excel(tif_file, excel_file)

print(f"Le fichier Excel a été créé avec succès : {excel_file}")
