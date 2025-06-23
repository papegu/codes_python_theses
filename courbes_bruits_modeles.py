import numpy as np
import matplotlib.pyplot as plt

# Niveaux de bruit
noise_levels = np.array([1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# MSE des modèles en fonction du bruit
mse_rnn = [0.0117, 0.0116, 0.0150, 0.0273, 0.0378, 0.0395, 0.0844, 0.1413, 0.2067, 0.3742, 0.3176, 0.3005]
mse_transformer = [0.0161, 0.0148, 0.0157, 0.0169, 0.0148, 0.0174, 0.0435, 0.0235, 0.0366, 0.0626, 0.0721, 0.0383]
mse_cnn_lstm = [0.0096, 0.0090, 0.0095, 0.0099, 0.0128, 0.0112, 0.0203, 0.0185, 0.0189, 0.0327, 0.0511, 0.0396]
mse_rtcnet = [0.0032, 0.0039, 0.0075, 0.0113, 0.0393, 0.0297, 0.0691, 0.1087, 0.1897, 0.2826, 0.2412, 0.2424]

# Tracé des courbes
plt.figure(figsize=(10, 6))
plt.plot(noise_levels, mse_rnn, marker='o', linestyle='-', label="RNN", color='blue')
plt.plot(noise_levels, mse_transformer, marker='s', linestyle='-', label="Transformer", color='red')
plt.plot(noise_levels, mse_cnn_lstm, marker='^', linestyle='-', label="CNN-LSTM", color='green')
plt.plot(noise_levels, mse_rtcnet, marker='d', linestyle='-', label="RTCNet", color='purple')

# Labels et titre
plt.xlabel("Noise Level (%)")
plt.ylabel("MSE")
plt.title("MSE of Models as a Function of Noise Level")
plt.legend()
plt.grid(True)

# Affichage du graphique
plt.show()
