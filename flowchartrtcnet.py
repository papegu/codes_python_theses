from graphviz import Digraph

# Créer un objet Digraph
dot = Digraph(comment='Architecture RTCNet-Hybrid', format='png')

# --- Définition des couleurs pour chaque type de couche ---
colors = {
    'InputLayer': 'lightgray',
    'Conv1D': 'lightblue',
    'MaxPooling1D': 'lightgreen',
    'SimpleRNN': 'lightcoral',
    'MultiHeadAttention': 'lightyellow',
    'GlobalAveragePooling1D': 'lightsalmon',
    'Concatenate': 'lightpink',
    'Dense': 'lightseagreen'
}

# --- Définition des noeuds de l'architecture ---
dot.node('InputLayer', 'InputLayer (Input)\n(None, 14, 1)', shape='box', style='filled', fillcolor=colors['InputLayer'])
dot.node('Conv1D', 'Conv1D (64 filters, kernel=2, ReLU)\n(None, 14, 64)\n192', shape='box', style='filled', fillcolor=colors['Conv1D'])
dot.node('MaxPooling1D', 'MaxPooling1D (pool=2)\n(None, 7, 64)', shape='box', style='filled', fillcolor=colors['MaxPooling1D'])
dot.node('SimpleRNN_1', 'SimpleRNN (64 units, ReLU, return_sequences=True)\n(None, 7, 64)\n8,256', shape='box', style='filled', fillcolor=colors['SimpleRNN'])
dot.node('SimpleRNN_2', 'SimpleRNN (64 units, ReLU, return_sequences=False)\n(None, 64)\n8,256', shape='box', style='filled', fillcolor=colors['SimpleRNN'])
dot.node('MultiHeadAttention', 'MultiHeadAttention (4 heads, key_dim=64)\n(None, 7, 64)\n66,368', shape='box', style='filled', fillcolor=colors['MultiHeadAttention'])
dot.node('GlobalAveragePooling1D', 'GlobalAveragePooling1D\n(None, 64)', shape='box', style='filled', fillcolor=colors['GlobalAveragePooling1D'])
dot.node('Concatenate', 'Concatenate\n(None, 128)', shape='box', style='filled', fillcolor=colors['Concatenate'])
dot.node('Dense_1', 'Dense (64 units, ReLU)\n(None, 64)\n8,256', shape='box', style='filled', fillcolor=colors['Dense'])
dot.node('Dense_2', 'Dense (3 units, Linear)\n(None, 3)\n195', shape='box', style='filled', fillcolor=colors['Dense'])

# --- Connexions entre les couches ---
dot.edge('InputLayer', 'Conv1D', label='Entrée vers\nConv1D')
dot.edge('Conv1D', 'MaxPooling1D', label='Conv1D vers\nMaxPooling1D')
dot.edge('MaxPooling1D', 'SimpleRNN_1', label='MaxPooling1D vers\nSimpleRNN')
dot.edge('SimpleRNN_1', 'SimpleRNN_2', label='SimpleRNN vers\nSimpleRNN')
dot.edge('MaxPooling1D', 'MultiHeadAttention', label='MaxPooling1D vers\nMultiHeadAttention')
dot.edge('MultiHeadAttention', 'GlobalAveragePooling1D', label='MultiHeadAttention vers\nGlobalAveragePooling1D')
dot.edge('SimpleRNN_2', 'Concatenate', label='SimpleRNN vers\nConcatenate')
dot.edge('GlobalAveragePooling1D', 'Concatenate', label='GlobalAveragePooling1D vers\nConcatenate')
dot.edge('Concatenate', 'Dense_1', label='Concatenate vers\nDense (ReLU)')
dot.edge('Dense_1', 'Dense_2', label='Dense (ReLU) vers\nDense (Linear)')

# --- Génération et affichage de l'organigramme ---
dot.render('architecture_rtcnet_hybrid', view=True)
