import requests
import pandas as pd
import zipfile
import io

# Définir l'URL pour l'API SSURGO (vous pouvez adapter la requête en fonction des régions géographiques ou des propriétés de sols spécifiques)
url = "https://sdmdataaccess.nrcs.usda.gov/Tabular/SDMTabularDownload.aspx"

# Requête pour télécharger les données SSURGO
params = {
    'data': 'ssurgo',  # Choisir 'ssurgo' pour télécharger les données SSURGO
    'download': 'True',
    'type': 'csv',  # Format CSV (ou 'zip' pour obtenir le fichier compressé)
    'state': 'US',  # Vous pouvez adapter selon la région géographique, ou en utilisant le code du pays
    # Ajoutez d'autres paramètres ici si nécessaire (comme le code de la région)
}

response = requests.get(url, params=params)

# Vérifier la réponse
if response.status_code == 200:
    # Si la réponse est valide, enregistrer le fichier zip
    print("Données téléchargées avec succès.")

    # Enregistrer le fichier zip sur le disque
    zip_file_path = 'soil_data.zip'
    with open(zip_file_path, 'wb') as f:
        f.write(response.content)

    # Extraire le fichier zip
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall('soil_data')

    print("Fichier zip extrait.")
    
    # Charger les fichiers extraits dans un DataFrame Pandas
    # Supposons qu'il y a un fichier CSV à l'intérieur
    csv_file_path = 'soil_data/soil_properties.csv'
    soil_data = pd.read_csv(csv_file_path)

    # Afficher les premières lignes pour vérifier
    print(soil_data.head())
else:
    print(f"Erreur de téléchargement des données. Code : {response.status_code}")

