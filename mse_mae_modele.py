import numpy as np
import matplotlib.pyplot as plt

# Données de performance
models = ["RNN", "Transformer", "CNN-LSTM", "RTCNet"]
mse_values = [0.018630, 0.012783, 0.013816, 0.001580]
mae_values = [0.094262, 0.070768, 0.076367, 0.029427]


# Largeur des barres
bar_width = 0.4
x = np.arange(len(models))

# Création de l'histogramme
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - bar_width/2, mse_values, bar_width, label="MSE", color='skyblue')
bars2 = ax.bar(x + bar_width/2, mae_values, bar_width, label="MAE", color='salmon')

# Ajouter des labels et un titre
ax.set_xlabel("Model")
ax.set_ylabel("Error Value")
ax.set_title("Performance Comparison of Models")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Afficher les valeurs sur les barres
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f"{height:.4f}", ha='center', va='bottom')

# Afficher le graphique
plt.show()
