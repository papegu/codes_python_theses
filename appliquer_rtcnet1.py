import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import geemap
import ee
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 🔹 Initialisation de Google Earth Engine
ee.Initialize()

# 🔹 Charger le modèle RTCNet
rtcnet_model = tf.keras.models.load_model(
    "RTCNet_model.h5",
    custom_objects={"mse": tf.keras.metrics.MeanSquaredError()}
)

# 🔹 Charger les données historiques
file_path = 'fertilite_senegal_biom.xlsx'
df = pd.read_excel(file_path)

# 🔹 Prétraitement des données
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces in column names
df.columns = df.columns.str.replace(" ", "_")  # Replace spaces with underscores in column names

# Ensure that the column 'Région' exists and map the region names to corresponding numbers
regions_dict = {'Dakar': 1, 'Thies': 2, 'Kaolack': 3, 'Saint-Louis': 4, 'Ziguinchor': 5, 
                'Diourbel': 6, 'Fatick': 7, 'Kaffrine': 8, 'Kolda': 9, 'Louga': 10, 
                'Matam': 11, 'Sédhiou': 12, 'Tambacounda': 13, 'Podor': 14}

# Check if 'Région' exists and map it to the appropriate region code
if 'Région' in df.columns:
    df["Région"] = df["Région"].map(regions_dict)
else:
    print("Column 'Région' not found in the dataset!")

# 🔹 Encodage de la colonne 'Culture_Adaptée'
encoder = OneHotEncoder(sparse_output=False)
df_encoded_culture = encoder.fit_transform(df[['Culture_Adaptée']])
df_encoded_culture = pd.DataFrame(df_encoded_culture, columns=encoder.categories_[0])

# Exclure les colonnes non numériques
numeric_columns = df.select_dtypes(include=[np.number]).columns
df = pd.concat([df[numeric_columns], df_encoded_culture], axis=1)

# 🔹 Normalisation des données
scaler_X = StandardScaler()
features = ['Année', 'Région', 'pH', 'Argile', 'Matière_Organique', 'Azote_(N)', 
            'Phosphore_(P)', 'Potassium_(K)', 'Latitude', 'Longitude'] + list(encoder.categories_[0])
X_scaled = scaler_X.fit_transform(df[features].values)

# 🔹 Récupération du NDVI à partir de Sentinel-2
def get_ndvi(lat, lon, year):
    """Récupère le NDVI moyen pour une localisation et une année donnée avec contrôles supplémentaires."""
    try:
        point = ee.Geometry.Point(lon, lat)
        collection = ee.ImageCollection("COPERNICUS/S2") \
            .filterBounds(point) \
            .filterDate(f"{year}-01-01", f"{year}-12-31") \
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10)) \
            .map(lambda img: img.normalizedDifference(["B8", "B4"]).rename("NDVI")) \
            .mean()

        # Calcul du NDVI
        ndvi = collection.reduceRegion(ee.Reducer.mean(), point, 30).get("NDVI").getInfo()

        # Vérifier si le NDVI est valide (entre -1 et 1)
        if ndvi is None or not (-1 <= ndvi <= 1):
            print(f"NDVI invalide pour ({lat}, {lon}) en {year}: {ndvi}")
            return 0  # Retourne 0 si le NDVI est invalide
        return ndvi
    except Exception as e:
        print(f"Erreur lors de la récupération du NDVI pour ({lat}, {lon}) en {year}: {e}")
        return 0  # Retourne 0 en cas d'erreur

# 🔹 Fonction d'estimation des propriétés à partir du NDVI
def estimate_properties_from_ndvi(ndvi):
    """Ajuste le NDVI pour estimer SOC, Biomasse et Fertilité dans la plage RTCNet attendue."""
    # Mise à l'échelle du NDVI de [0, 0.5] à [1.5, 3.5] (plage RTCNet)
    Fertilité = 4 * ndvi + 1.5  # Mise à l'échelle linéaire du NDVI pour correspondre à la plage de fertilité RTCNet [1.5, 3.5]
    
    # Estimations supplémentaires (si nécessaire)
    SOC = 1.5 * ndvi + 0.5
    Biomasse = 2.8 * ndvi + 1.2
    
    return np.array([SOC, Biomasse, Fertilité])

# 🔹 Prédiction avec RTCNet
def predict_future_properties(model, df, year):
    """Prédit SOC, Biomasse et Fertilité pour une année future."""
    past_years = df[df["Année"].isin([year - 1, year - 2])]
    future_input = past_years.mean(axis=0)
    future_input["Année"] = year

    # Vérifiez que le nombre de caractéristiques dans 'X_future' est correct
    X_future = future_input[features].values.reshape(1, -1)  # Shape: (1, 12)
    
    # Appliquer la normalisation
    X_future_scaled = scaler_X.transform(X_future)  # Appliquer la normalisation

    # Si le modèle attend 14 caractéristiques, assurez-vous que X_future_scaled a la forme (1, 14)
    if X_future_scaled.shape[1] != 14:
        print(f"Avertissement: X_future_scaled a {X_future_scaled.shape[1]} caractéristiques au lieu de 14.")
        return None  # Retourner None si la forme est incorrecte

    # Redimensionner pour correspondre à l'entrée attendue par RTCNet (1, 14, 1)
    X_future_reshaped = X_future_scaled.reshape((1, 14, 1)).astype(np.float32)

    # Prédiction
    y_future_pred = model.predict(X_future_reshaped)
    return y_future_pred[0]

# 🔹 Créer un dictionnaire pour stocker les résultats de chaque région
regions_fertility = {}

regions = df["Région"].unique()
years = [2025]

for region in regions:
    print(f"Processing predictions for region {region}...")
    
    df_region = df[df["Région"] == region]
    locations = df_region[["Latitude", "Longitude"]].drop_duplicates().values
    
    rtcnet_preds = []
    ndvi_preds = []

    for lat, lon in locations:
        # Prédictions RTCNet
        pred_rtcnet = predict_future_properties(rtcnet_model, df_region, years[0])
        
        # NDVI & estimation des propriétés du sol
        ndvi_value = get_ndvi(lat, lon, years[0])
        pred_ndvi = estimate_properties_from_ndvi(ndvi_value)[2]  # Fertilité (indice 2)
        
        rtcnet_preds.append(pred_rtcnet[2])  # Fertilité prédite par RTCNet (indice 2)
        ndvi_preds.append(pred_ndvi)

    # Ajouter les résultats dans le dictionnaire, en utilisant la région comme clé
    regions_fertility[region] = {
        "latitude": locations[:, 0],  # Collecte des latitudes uniques
        "rtcnet_fertility": np.array(rtcnet_preds),
        "ndvi_fertility": np.array(ndvi_preds)
    }

# 🔹 Calcul des erreurs MAE et MSE
mae_list = []
mse_list = []
regions_list = list(regions_fertility.keys())

for region in regions_list:
    rtcnet_fertility = regions_fertility[region]["rtcnet_fertility"]
    ndvi_fertility = regions_fertility[region]["ndvi_fertility"]
    
    # Calculer MAE et MSE
    mae = mean_absolute_error(rtcnet_fertility, ndvi_fertility)
    mse = mean_squared_error(rtcnet_fertility, ndvi_fertility)
    
    mae_list.append(mae)
    mse_list.append(mse)

# 🔹 Tracer les résultats de MAE et MSE
plt.figure(figsize=(12, 6))
x = np.arange(len(regions_fertility))

# Barres pour MAE et MSE
plt.bar(x - 0.2, mae_list, width=0.4, label="MAE", color='blue')
plt.bar(x + 0.2, mse_list, width=0.4, label="MSE", color='red')

# Ajouter des labels et titre
plt.xlabel('Region')
plt.ylabel('Error')
plt.title('MAE and MSE Comparison by Region')

# Placer les noms des régions en abscisses
plt.xticks(x, list(regions_fertility.keys()), rotation=45)

plt.legend()
plt.tight_layout()
plt.show()

# 🔹 Afficher l'histogramme des différences
fertility_differences = []
for region in regions_list:
    rtcnet_fertility = regions_fertility[region]["rtcnet_fertility"]
    ndvi_fertility = regions_fertility[region]["ndvi_fertility"]
    
    # Calculer la différence absolue entre les fertilités RTCNet et NDVI
    fertility_differences.extend(np.abs(rtcnet_fertility - ndvi_fertility))  # Stocke les différences

plt.figure(figsize=(10, 6))
plt.hist(fertility_differences, bins=20, color='purple', alpha=0.7)
plt.xlabel('Absolute Difference in Fertility')
plt.ylabel('Frequency')
plt.title('Histogram of Absolute Differences Between RTCNet and NDVI Fertility')
plt.tight_layout()
plt.show()
