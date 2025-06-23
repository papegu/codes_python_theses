import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras import regularizers

# --- Chargement et prétraitement des données ---

file_path = file_path = "dataset12mai.xlsx"

df = pd.read_excel(file_path)

# Nettoyage noms colonnes
df.columns = df.columns.str.strip().str.replace(" ", "_")

region_mapping = {
    'Dakar': 1, 'Thies': 2, 'Kaolack': 3, 'Saint-Louis': 4, 'Ziguinchor': 5,
    'Diourbel': 6, 'Fatick': 7, 'Kaffrine': 8, 'Kolda': 9, 'Louga': 10,
    'Matam': 11, 'Sédhiou': 12, 'Tambacounda': 13, 'Podor': 14
}
df['Région'] = df['Région'].map(region_mapping)

encoder = OneHotEncoder(sparse_output=False)
encoded_culture = encoder.fit_transform(df[['Culture_Adaptée']])
culture_columns = encoder.categories_[0]
df_encoded_culture = pd.DataFrame(encoded_culture, columns=culture_columns)
df = pd.concat([df, df_encoded_culture], axis=1)

X = df[['Année', 'Région', 'pH', 'Argile', 'Matière_Organique', 'Azote_(N)', 'Phosphore_(P)', 'Potassium_(K)',
        'Latitude', 'Longitude'] + list(culture_columns)].values
y = df[['SOC', 'Biomasse', 'Fertilité']].values

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)).astype(np.float32)
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1)).astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# --- Fonctions de visualisation ---
def plot_training_history(history, model_name="Modèle"):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Épochs')
    plt.ylabel('Loss')
    plt.legend()
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
    plt.figure(figsize=(14, 8))
    for i, prop in enumerate(['SOC', 'Biomasse', 'Fertilité']):
        plt.subplot(2, 3, i + 1)
        plt.plot(y_true[:, i], label='Réel', color='blue')
        plt.plot(y_pred[:, i], label='Prévu', color='red')
        plt.title(f"Propriété: {prop}")
        plt.xlabel("Échantillons")
        plt.ylabel("Valeur")
        plt.legend()
    plt.tight_layout()
    plt.suptitle(f"Comparaison des prédictions pour {model_name}", fontsize=12)
    plt.show()

# --- Modèles ---

def create_rnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(3, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_transformer_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(64, kernel_size=2, activation='relu')(inputs)
    x = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(3, activation='linear')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_cnn_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, kernel_size=2, activation='relu', input_shape=input_shape),
        tf.keras.layers.LSTM(50, return_sequences=True),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(3, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_rtcnet_hybrid_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    cnn = tf.keras.layers.Conv1D(64, kernel_size=2, activation='relu', padding='same',
                                 kernel_regularizer=regularizers.l2(0.01))(inputs)
    cnn = tf.keras.layers.MaxPooling1D(pool_size=2)(cnn)
    cnn = tf.keras.layers.Dropout(0.3)(cnn)

    rnn = tf.keras.layers.SimpleRNN(64, activation='relu', return_sequences=True,
                                    kernel_regularizer=regularizers.l2(0.01))(cnn)
    rnn = tf.keras.layers.SimpleRNN(64, activation='relu', return_sequences=False)(rnn)
    rnn = tf.keras.layers.Dropout(0.3)(rnn)

    transformer = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(cnn, cnn)
    transformer = tf.keras.layers.GlobalAveragePooling1D()(transformer)
    transformer = tf.keras.layers.Dropout(0.3)(transformer)

    concatenated = tf.keras.layers.concatenate([rnn, transformer])

    dense = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(concatenated)
    outputs = tf.keras.layers.Dense(3, activation='linear')(dense)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_ann_model(input_shape):
    # ANN classique : entrée plate, donc input_shape=(nombre de features,)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_transformer_cnn_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
    x = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(3, activation='linear')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_rtcnet_improved_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    cnn = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same',
                                 kernel_regularizer=regularizers.l2(0.01))(inputs)
    cnn = tf.keras.layers.MaxPooling1D(pool_size=2)(cnn)
    cnn = tf.keras.layers.Dropout(0.3)(cnn)

    bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(cnn)
    bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False))(bi_lstm)
    bi_lstm = tf.keras.layers.Dropout(0.3)(bi_lstm)

    mha = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(cnn, cnn)
    mha = tf.keras.layers.GlobalAveragePooling1D()(mha)
    mha = tf.keras.layers.Dropout(0.3)(mha)

    concat = tf.keras.layers.concatenate([bi_lstm, mha])

    dense = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(concat)
    dense = tf.keras.layers.Dropout(0.3)(dense)
    outputs = tf.keras.layers.Dense(3, activation='linear')(dense)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# --- Préparer les modèles ---

models = {
    "RNN": create_rnn_model((X_train_reshaped.shape[1], 1)),
    "Transformer": create_transformer_model((X_train_reshaped.shape[1], 1)),
    "CNN-LSTM": create_cnn_lstm_model((X_train_reshaped.shape[1], 1)),
    "RTCNet": create_rtcnet_hybrid_model((X_train_reshaped.shape[1], 1)),
    "ANN": create_ann_model(X_train.shape[1]),
    "Transformer+CNN": create_transformer_cnn_model((X_train_reshaped.shape[1], 1)),
    "RTCNet Improved": create_rtcnet_improved_model((X_train_reshaped.shape[1], 1))
}

performance = {"Model": [], "MSE": [], "MAE": [], "MSE_with_noise": [], "MAE_with_noise": []}

# --- Entraînement, évaluation et test robustesse bruit ---

noise_levels = [0.1]  # on va tester 10% de bruit ajouté pour comparaison simple

for model_name, model in models.items():
    print(f"\n=== Entraînement du modèle : {model_name} ===")
    
    # Pour ANN, on utilise les données plates, sinon données reshaped
    if model_name == "ANN":
        history = model.fit(X_train, y_train, epochs=50, batch_size=16,
                            validation_data=(X_test, y_test), verbose=0)
    else:
        history = model.fit(X_train_reshaped, y_train, epochs=50, batch_size=16,
                            validation_data=(X_test_reshaped, y_test), verbose=0)

    plot_training_history(history, model_name)

    # Prédictions sans bruit
    if model_name == "ANN":
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test_reshaped)

    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))

    # Prédictions avec bruit
    if model_name == "ANN":
        X_test_noisy = X_test + noise_levels[0] * np.random.normal(size=X_test.shape)
        y_pred_noisy = model.predict(X_test_noisy)
    else:
        X_test_noisy = X_test_reshaped + noise_levels[0] * np.random.normal(size=X_test_reshaped.shape)
        y_pred_noisy = model.predict(X_test_noisy)

    mse_noisy = np.mean((y_test - y_pred_noisy) ** 2)
    mae_noisy = np.mean(np.abs(y_test - y_pred_noisy))

    performance["Model"].append(model_name)
    performance["MSE"].append(mse)
    performance["MAE"].append(mae)
    performance["MSE_with_noise"].append(mse_noisy)
    performance["MAE_with_noise"].append(mae_noisy)

    print(f"{model_name} - MSE (sans bruit): {mse:.4f}, MAE (sans bruit): {mae:.4f}")
    print(f"{model_name} - MSE (avec bruit 10%): {mse_noisy:.4f}, MAE (avec bruit 10%): {mae_noisy:.4f}")

    plot_predictions(y_test, y_pred, model_name=model_name)

performance_df = pd.DataFrame(performance)
print("\n=== Résumé des performances ===")
print(performance_df)
# --- Sous-modèles pour étude d'ablation ---

def rtcnet_no_bilstm(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    cnn = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same',
                                 kernel_regularizer=regularizers.l2(0.01))(inputs)
    cnn = tf.keras.layers.MaxPooling1D(pool_size=2)(cnn)
    cnn = tf.keras.layers.Dropout(0.3)(cnn)

    mha = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(cnn, cnn)
    mha = tf.keras.layers.GlobalAveragePooling1D()(mha)
    mha = tf.keras.layers.Dropout(0.3)(mha)

    dense = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(mha)
    dense = tf.keras.layers.Dropout(0.3)(dense)
    outputs = tf.keras.layers.Dense(3, activation='linear')(dense)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def rtcnet_no_attention(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    cnn = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same',
                                 kernel_regularizer=regularizers.l2(0.01))(inputs)
    cnn = tf.keras.layers.MaxPooling1D(pool_size=2)(cnn)
    cnn = tf.keras.layers.Dropout(0.3)(cnn)

    bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(cnn)
    bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False))(bi_lstm)
    bi_lstm = tf.keras.layers.Dropout(0.3)(bi_lstm)

    dense = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(bi_lstm)
    dense = tf.keras.layers.Dropout(0.3)(dense)
    outputs = tf.keras.layers.Dense(3, activation='linear')(dense)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def rtcnet_no_cnn(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputs)
    bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False))(bi_lstm)
    bi_lstm = tf.keras.layers.Dropout(0.3)(bi_lstm)

    mha = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
    mha = tf.keras.layers.GlobalAveragePooling1D()(mha)
    mha = tf.keras.layers.Dropout(0.3)(mha)

    concat = tf.keras.layers.concatenate([bi_lstm, mha])
    dense = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(concat)
    dense = tf.keras.layers.Dropout(0.3)(dense)
    outputs = tf.keras.layers.Dense(3, activation='linear')(dense)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

ablation_models = {
    "RTCNet Improved": create_rtcnet_improved_model((X_train_reshaped.shape[1], 1)),
    "No BiLSTM": rtcnet_no_bilstm((X_train_reshaped.shape[1], 1)),
    "No Attention": rtcnet_no_attention((X_train_reshaped.shape[1], 1)),
    "No CNN": rtcnet_no_cnn((X_train_reshaped.shape[1], 1))
}

ablation_performance = {"Model": [], "MSE": [], "MAE": [], "MSE_with_noise": [], "MAE_with_noise": []}
print(">>> Début de l’étude d’ablation <<<")

for model_name, model in ablation_models.items():
    print(f"\n=== Étude d’ablation : {model_name} ===")
    history = model.fit(X_train_reshaped, y_train, epochs=50, batch_size=16,
                        validation_data=(X_test_reshaped, y_test), verbose=0)

    y_pred = model.predict(X_test_reshaped)
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))

    # Ajout de bruit
    X_test_noisy = X_test_reshaped + 0.1 * np.random.normal(size=X_test_reshaped.shape)
    y_pred_noisy = model.predict(X_test_noisy)
    mse_noisy = np.mean((y_test - y_pred_noisy) ** 2)
    mae_noisy = np.mean(np.abs(y_test - y_pred_noisy))

    ablation_performance["Model"].append(model_name)
    ablation_performance["MSE"].append(mse)
    ablation_performance["MAE"].append(mae)
    ablation_performance["MSE_with_noise"].append(mse_noisy)
    ablation_performance["MAE_with_noise"].append(mae_noisy)

    print(f"{model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}")
    print(f"{model_name} - MSE avec bruit: {mse_noisy:.4f}, MAE avec bruit: {mae_noisy:.4f}")
