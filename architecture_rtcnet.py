import tensorflow as tf
from tensorflow.keras.utils import plot_model

def create_rtcnet_hybrid_model(input_shape):
    # Entrée
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

    return model

# Création du modèle
input_shape = (13, 1)  # Exemple de séquence temporelle de longueur 100 avec une seule variable
model = create_rtcnet_hybrid_model(input_shape)

# Générer un schéma du modèle
plot_model(model, to_file="rtcnet_hybrid_model.png", show_shapes=True, show_layer_names=True)

# Affichage de l'architecture
model.summary()
