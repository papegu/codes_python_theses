import tensorflow as tf
from tensorflow.keras.utils import plot_model

# Définir la fonction du modèle RTCNet Hybrid
def create_rtcnet_hybrid_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # --- Bloc CNN ---
    cnn = tf.keras.layers.Conv1D(64, kernel_size=2, activation='relu', padding='same')(inputs)
    cnn = tf.keras.layers.MaxPooling1D(pool_size=2)(cnn)

    # --- Bloc RNN ---
    rnn = tf.keras.layers.SimpleRNN(64, activation='relu', return_sequences=True)(cnn)
    rnn = tf.keras.layers.SimpleRNN(64, activation='relu', return_sequences=False)(rnn)

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

# Définir la forme d'entrée (16, 14, 1)
input_shape = (14, 1)

# Créer le modèle
model = create_rtcnet_hybrid_model(input_shape)

# Afficher l'architecture sous forme de graphique
plot_model(model, to_file='rtcnet_hybrid_model.png', show_shapes=True, show_layer_names=True)
