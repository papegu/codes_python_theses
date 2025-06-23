import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

# Charger les données
file_path = 'fertilite_senegal_biom.xlsx'
df = pd.read_excel(file_path)
# Nettoyage des noms de colonnes
df.columns = df.columns.str.strip()  # Supprime les espaces en début et fin
df.columns = df.columns.str.replace(" ", "_")  # Remplace les espaces internes par des underscores

#print(df.columns.tolist())  # Pour confirmer après nettoyage

print(df.columns.tolist())


# Liste des régions avec leur correspondance numérique
region_mapping = {
    'Dakar': 1, 'Thies': 2, 'Kaolack': 3, 'Saint-Louis': 4, 'Ziguinchor': 5, 
    'Diourbel': 6, 'Fatick': 7, 'Kaffrine': 8, 'Kolda': 9, 'Louga': 10, 
    'Matam': 11, 'Sédhiou': 12, 'Tambacounda': 13, 'Podor': 14
}

# Remplacer la colonne Région par les numéros correspondants
df['Région'] = df['Région'].map(region_mapping)

# Appliquer OneHotEncoder sur la colonne 'Culture Adaptée'

encoder = OneHotEncoder(sparse_output=False)

#encoded_culture = encoder.fit_transform(df[['Culture Adaptée']])
encoded_culture = encoder.fit_transform(df[['Culture_Adaptée']])


# Ajouter les nouvelles colonnes one-hot encodées dans le DataFrame
culture_columns = encoder.categories_[0]  # Nom des catégories de Culture Adaptée
df_encoded_culture = pd.DataFrame(encoded_culture, columns=culture_columns)

# Fusionner les colonnes encodées dans le DataFrame principal
df = pd.concat([df, df_encoded_culture], axis=1)

# Sélectionner les entrées et sorties
X = df[['Année', 'Région', 'pH', 'Argile', 'Matière Organique', 'Azote (N)', 'Phosphore (P)', 'Potassium (K)', 
        'Latitude', 'Longitude'] + list(culture_columns)].values

# La colonne "Culture Adaptée" ne doit pas faire partie des cibles (y)
#y = df[['SOC', 'Biomasse', 'Fertilité']].values  # Exclure "Culture Adaptée" ici
y = df[['SOC', 'Biomasse', 'Fertilité']].values

# Normalisation des données d'entrée
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshaping pour les modèles séquentiels
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)).astype(np.float32)
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1)).astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# Extraire les années pour les graphiques
years = df['Année'].values

# --- Fonctions d'évaluation et de visualisation ---
def plot_training_history(history, model_name="Modèle"):
    """Trace les courbes d'entraînement et de validation."""
    plt.figure(figsize=(12, 6))
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Épochs')
    plt.ylabel('Loss')
    plt.legend()
    # MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'{model_name} - MAE')
    plt.xlabel('Épochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_predictions(y_true, y_pred, model_name="Modèle"):
    """Affiche une comparaison graphique des valeurs réelles et prédites."""
    plt.figure(figsize=(14, 8))
    for i, prop in enumerate(['SOC', 'Biomasse', 'Fertilité']):
        plt.subplot(2, 3, i + 1)
        plt.plot(years[-len(y_true):], y_true[:, i], label='Real', color='blue')
        plt.plot(years[-len(y_true):], y_pred[:, i], label='Forecast', color='red')
        plt.title(f"Property: {prop}")
        plt.xlabel("Years")
        plt.ylabel("Value")
        plt.legend()
    plt.tight_layout()
    #plt.suptitle(f"Comparison of predictions for {model_name}", fontsize=10)
    plt.show()

# --- Création des modèles ---
def create_rnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(64, activation='relu', input_shape=input_shape, return_sequences=False),
        tf.keras.layers.Dense(3, activation='linear')  # 3 cibles
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_transformer_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(64, kernel_size=2, activation='relu')(inputs)
    x = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(3, activation='linear')(x)  # 3 cibles
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_cnn_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, kernel_size=2, activation='relu', input_shape=input_shape),
        tf.keras.layers.LSTM(50, return_sequences=True),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(3, activation='linear')  # 3 cibles
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_rtcnet_hybrid_model(input_shape):
    # Entrée
    inputs = tf.keras.Input(shape=input_shape)

    # --- Bloc CNN ---
    cnn = tf.keras.layers.Conv1D(64, kernel_size=2, activation='relu', padding='same')(inputs)
    cnn = tf.keras.layers.MaxPooling1D(pool_size=2)(cnn)

    # --- Bloc RNN ---
    rnn = tf.keras.layers.LSTM(64, activation='relu', return_sequences=True)(cnn)
    rnn = tf.keras.layers.LSTM(64, activation='relu', return_sequences=False)(rnn)

    # --- Bloc Transformer ---
    transformer = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(cnn, cnn)
    transformer = tf.keras.layers.GlobalAveragePooling1D()(transformer)

    # --- Fusion des sorties des blocs ---
    concatenated = tf.keras.layers.concatenate([rnn, transformer])

    # --- Couches de sortie ---
    dense = tf.keras.layers.Dense(64, activation='relu')(concatenated)
    outputs = tf.keras.layers.Dense(3, activation='linear')(dense)  # 3 cibles

    # Création du modèle
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compilation du modèle
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model
print("Forme des tenseurs en entrée :", X_train_reshaped.shape)

# --- Entraînement et comparaison des modèles ---
models = {
    "RNN": create_rnn_model((X_train_reshaped.shape[1], 1)),
    "Transformer": create_transformer_model((X_train_reshaped.shape[1], 1)),
    "CNN-LSTM": create_cnn_lstm_model((X_train_reshaped.shape[1], 1)),
    "RTCNet": create_rtcnet_hybrid_model((X_train_reshaped.shape[1], 1))
}

performance = {"Model": [], "MSE": [], "MAE": []}

for model_name, model in models.items():
    print(f"Model training : {model_name}")
    history = model.fit(
        X_train_reshaped, y_train, epochs=50, batch_size=16, validation_data=(X_test_reshaped, y_test), verbose=0
    )
    # Tracer les courbes d'entraînement
    plot_training_history(history, model_name=model_name)
    
    # Évaluer le modèle
    y_pred = model.predict(X_test_reshaped)
    mse = np.mean((y_test - y_pred)**2)
    mae = np.mean(np.abs(y_test - y_pred))
    
    # Stocker les résultats
    performance["Model"].append(model_name)
    performance["MSE"].append(mse)
    performance["MAE"].append(mae)
    
    # Tracer les prédictions
    plot_predictions(y_test, y_pred, model_name=model_name)

# Affichage des performances
performance_df = pd.DataFrame(performance)
print("\nPerformance comparison:")
print(performance_df)

# --- Test de robustesse au bruit ---
noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
print("\nAssessment of noise robustness :")
for noise in noise_levels:
    print(f"\n--- Bruit : {noise * 100:.1f}% ---")
    X_test_noisy = X_test_reshaped + noise * np.random.normal(size=X_test_reshaped.shape)
    for model_name, model in models.items():
        y_pred_noisy = model.predict(X_test_noisy)
        mse_noisy = np.mean((y_test - y_pred_noisy)**2)
        print(f"{model_name} -  MSE with noise : {mse_noisy:.4f}")
