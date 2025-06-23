from osgeo import ogr
import pandas as pd

# Fonction pour extraire les données d'une couche de la GDB et les convertir en DataFrame
def gdb_to_dataframe(gdb_path, layer_name):
    # Charger la GDB
    driver = ogr.GetDriverByName('OpenFileGDB')
    if not driver:
        raise Exception("Driver GDB non trouvé")
    
    # Ouvrir la Geodatabase
    gdb = driver.Open(gdb_path, 0)
    if not gdb:
        raise Exception("Impossible d'ouvrir la Geodatabase")
    
    # Accéder à la couche spécifiée
    layer = gdb.GetLayerByName(layer_name)
    if not layer:
        raise Exception(f"La couche {layer_name} n'a pas été trouvée dans la GDB")
    
    # Liste des champs (colonnes)
    field_names = [field.name for field in layer.schema]
    
    # Extraire les données de la couche
    data = []
    for feature in layer:
        row = []
        for field_name in field_names:
            row.append(feature.GetField(field_name))
        data.append(row)
    
    # Créer un DataFrame pandas
    df = pd.DataFrame(data, columns=field_names)
    
    return df

# Fonction pour exporter un DataFrame vers un fichier Excel (.xlsx)
def export_to_excel(df, output_file):
    df.to_excel(output_file, index=False, engine='openpyxl')

# Exemple d'utilisation
gdb_file = 'path_to_your_file.gdb'  # Chemin vers le fichier .gdb
layer_name = 'nom_de_la_couche'     # Nom de la couche dans la GDB
output_file = 'output.xlsx'         # Nom du fichier Excel de sortie

# Extraire les données de la GDB dans un DataFrame
df = gdb_to_dataframe(gdb_file, layer_name)

# Exporter les données vers un fichier Excel
export_to_excel(df, output_file)

print(f"Les données ont été exportées avec succès vers {output_file}")
