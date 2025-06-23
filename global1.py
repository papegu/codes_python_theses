from graphviz import Digraph

# Création du graphe
dot = Digraph("RTCNet_Flow", format="png")

# Définition des styles des nœuds
dot.attr("node", shape="box", style="filled", fillcolor="#DDEEFF", fontname="Arial", fontsize="12")

# Entrée
dot.node("Input", "Input (16, 14, 1)", fillcolor="#FFDDC1")

# Bloc CNN
dot.node("Conv1D", "Conv1D (64 filters, kernel=2) \n(16, 13, 64)")
dot.node("MaxPool", "MaxPooling1D (pool=2) \n(16, 6, 64)")

# Bloc RNN
dot.node("RNN", "SimpleRNN (64 units) \n(16, 6, 64)")

# Bloc Transformer
dot.node("MHA", "MultiHeadAttention (4 heads) \n(16, 6, 64)")

# Fusion et Agrégation
dot.node("GlobalPool_RNN", "GlobalAvgPooling1D (RNN) \n(16, 64)")
dot.node("GlobalPool_MHA", "GlobalAvgPooling1D (MHA) \n(16, 64)")
dot.node("Concat", "Concatenation \n(16, 128)")

# Couches Denses finales
dot.node("Dense1", "Dense (64 units, ReLU) \n(16, 64)")
dot.node("Dense2", "Dense (3 units, Linear) \n(16, 3)", fillcolor="#FFCCCC")

# Connexions entre les couches
dot.edge("Input", "Conv1D")
dot.edge("Conv1D", "MaxPool")
dot.edge("MaxPool", "RNN")
dot.edge("MaxPool", "MHA")
dot.edge("RNN", "GlobalPool_RNN")
dot.edge("MHA", "GlobalPool_MHA")
dot.edge("GlobalPool_RNN", "Concat")
dot.edge("GlobalPool_MHA", "Concat")
dot.edge("Concat", "Dense1")
dot.edge("Dense1", "Dense2")

# Génération et affichage du graphe
dot.render("RTCNet_Flow", view=True)
