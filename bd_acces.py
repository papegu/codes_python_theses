import os
import pandas as pd
import pyodbc
from glob import glob

def get_access_file(folder_path):
    """Récupère le chemin du premier fichier Access trouvé dans un dossier donné."""
    files = glob(os.path.join(folder_path, "*.mdb")) + glob(os.path.join(folder_path, "*.accdb"))
    return files[0] if files else None

def export_access_to_excel(folder_path, output_excel):
    """Extrait les données d'une base Access et les exporte dans un fichier Excel unique."""
    access_file = get_access_file(folder_path)
    if not access_file:
        print("Aucun fichier Access trouvé.")
        return
    
    # Déterminer la chaîne de connexion
    if access_file.endswith(".mdb"):
        conn_str = f'DRIVER={{Microsoft Access Driver (*.mdb)}};DBQ={access_file};'
    else:
        conn_str = f'DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={access_file};'
    
    # Connexion à la base Access
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    # Récupération des noms de tables
    tables = [row.table_name for row in cursor.tables(tableType='TABLE')]
    
    all_data = []
    all_columns = set()
    
    # Lire toutes les tables et les concaténer
    for table in tables:
        df = pd.read_sql(f'SELECT * FROM [{table}]', conn)
        df['Table_Name'] = table  # Ajouter la colonne source
        all_data.append(df)
        all_columns.update(df.columns)
    
    # Créer un DataFrame unifié
    unified_df = pd.concat(all_data, ignore_index=True, sort=False)
    
    # Réorganiser les colonnes pour garantir l'ordre
    unified_df = unified_df.reindex(columns=sorted(all_columns))
    
    # Exporter vers Excel
    unified_df.to_excel(output_excel, index=False)
    print(f"Données exportées vers {output_excel}")
    
    conn.close()

# Exemple d'utilisation
folder_path = "C:/chemin/vers/dossier"  # Remplacez par le chemin réel
output_excel = "C:/chemin/vers/output.xlsx"  # Remplacez par le chemin souhaité
export_access_to_excel(folder_path, output_excel)
