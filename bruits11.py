import matplotlib.pyplot as plt

# Liste des niveaux de bruit
noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Résultats de MSE pour chaque modèle (remplissez les MSE obtenus pour chaque bruit et modèle)
mse_results = {
    "RNN": [0.0085, 0.0098, 0.0099, 0.0147, 0.0373, 0.0509, 0.0864, 0.1088, 0.1030, 0.1630, 0.2496, 0.4469],
    "Transformer": [0.0145, 0.0159, 0.0163, 0.0178, 0.0245, 0.0246, 0.0370, 0.0377, 0.0424, 0.0435, 0.0649, 0.1125],
    "CNN-LSTM": [0.0084, 0.0087, 0.0101, 0.0107, 0.0128, 0.0118, 0.0125, 0.0133, 0.0149, 0.0221, 0.0192, 0.0214],
    "RTCNet": [0.0167, 0.0171, 0.0160, 0.0197, 0.0184, 0.0196, 0.0187, 0.0225, 0.0176, 0.0348, 0.0481, 0.0523]
}

# Tracer les courbes de MSE en fonction du bruit pour chaque modèle
plt.figure(figsize=(10, 6))

for model_name, mse_values in mse_results.items():
    plt.plot(noise_levels, mse_values, label=model_name)

# Ajouter des labels et une légende
plt.xlabel("Noise level (%)")
plt.ylabel("MSE")
plt.title("Comparison of noise resistance of models")
plt.legend()
plt.grid(True)

# Afficher le graphique
plt.tight_layout()
plt.show()
