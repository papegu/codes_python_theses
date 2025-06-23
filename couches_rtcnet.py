import matplotlib.pyplot as plt
import numpy as np

# Fonction pour dessiner les neurones d'une couche avec des connexions entre elles
def draw_layer(ax, x_pos, num_neurons, layer_name, colors, layer_index, num_neurons_to_draw=5):
    # Limiter le nombre de neurones dessinés en fonction du nombre de neurones de la couche
    num_neurons_to_draw = min(num_neurons, num_neurons_to_draw)
    y_pos = np.linspace(0, num_neurons-1, num_neurons_to_draw)
    
    for i, y in enumerate(y_pos):
        ax.add_patch(plt.Circle((x_pos, y), 0.1, color=colors[i], ec='black', lw=2))

    # Ajouter des pointillés pour les neurones non dessinés (si le nombre de neurones est supérieur à 5)
    if num_neurons > num_neurons_to_draw:
        y_pos_dashed = np.linspace(0, num_neurons-1, num_neurons - num_neurons_to_draw)
        for y in y_pos_dashed:
            ax.plot([x_pos, x_pos], [y, y], 'k--', lw=1)

    # Ajouter des labels sous le premier neurone
    ax.text(x_pos, -1.5, layer_name, horizontalalignment='center', verticalalignment='center', fontsize=12, fontweight='bold')
    
    return y_pos

# Fonction pour ajouter les flèches entre les couches
def draw_connections(ax, y_pos_input, y_pos_output, x_pos_input, x_pos_output, color):
    for y_in in y_pos_input:
        for y_out in y_pos_output:
            ax.annotate('', xy=(x_pos_output, y_out), xytext=(x_pos_input, y_in), 
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.5))

# Initialisation de la figure
fig, ax = plt.subplots(figsize=(12, 8))

# Dimensions et couleurs des couches
layer_specs = [
    ('Input Layer', 10, ['green'] * 5, 0),  # Input, 5 neurons
    ('Conv1D (64 filters)', 64, ['blue'] * 5, 1),  # Conv1D, 5 neurons
    ('MaxPooling1D', 64, ['yellow'] * 5, 2),  # MaxPooling1D, 5 neurons
    ('SimpleRNN (64 units)', 64, ['pink'] * 5, 3),  # SimpleRNN, 5 neurons
    ('MultiHeadAtt(4heads)', 64, ['red'] * 5, 4),  # MultiHeadAttention, 5 neurons
    ('Dense(64 units)', 64, ['lightgray'] * 5, 5),  # Dense, 5 neurons
    ('OutputLayer(3 targets)', 3, ['black'] * 3, 6)  # Output, 3 neurons
]

# Dessiner chaque couche et ses connexions
y_pos_input = draw_layer(ax, 0, layer_specs[0][1], layer_specs[0][0], layer_specs[0][2], 0)
for i in range(1, len(layer_specs)):
    y_pos_output = draw_layer(ax, i, layer_specs[i][1], layer_specs[i][0], layer_specs[i][2], i)
    
    # Utiliser la couleur de la couche pour la flèche entre les couches
    connection_color = layer_specs[i][2][0]  # Récupérer la couleur de la couche (première couleur de la liste)
    draw_connections(ax, y_pos_input, y_pos_output, i-1, i, color=connection_color)
    
    y_pos_input = y_pos_output

# Réglages pour rendre le diagramme plus propre
ax.set_xlim(-0.5, len(layer_specs)-0.5)
ax.set_ylim(-2, 66)
ax.axis('off')

# Ajouter des légendes en bas
layer_names = [layer[0] for layer in layer_specs]

# Créer la légende en bas
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color[0], markersize=10) for color in [layer[2] for layer in layer_specs]]
ax.legend(handles, layer_names, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=12)

# Titre
plt.title("Simplified Architecture of the RTCNet Model", fontsize=16)
plt.tight_layout()
plt.show()
