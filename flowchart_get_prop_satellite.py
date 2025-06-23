from graphviz import Digraph

# Création du diagramme
flowchart = Digraph(format='png')
flowchart.attr(size='10')

# Définition des nœuds
flowchart.node('A', 'Start', shape='oval')
flowchart.node('B', 'Initialization of Google Earth Engine', shape='parallelogram')
flowchart.node('C', "Definition of Senegal's 14 regions", shape='parallelogram')
flowchart.node('D', 'Loop over the years (2018-2024)', shape='diamond')
flowchart.node('E', 'Load Sentinel-2 and calculate NDVI', shape='parallelogram')
flowchart.node('F', 'Valid NDVI check?', shape='diamond')
flowchart.node('G', 'Calculation of soil properties', shape='parallelogram')
flowchart.node('H', 'Storing results', shape='parallelogram')
flowchart.node('I', 'Moving on to the next year', shape='parallelogram')
flowchart.node('J', 'Backup in Excel', shape='parallelogram')
flowchart.node('K', 'End', shape='oval')

# Connexions
flowchart.edge('A', 'B')
flowchart.edge('B', 'C')
flowchart.edge('C', 'D')
flowchart.edge('D', 'E', label="For each year")
flowchart.edge('E', 'F')
flowchart.edge('F', 'G', label="Yes")
flowchart.edge('F', 'I', label="No")
flowchart.edge('G', 'H')
flowchart.edge('H', 'I')
flowchart.edge('I', 'D', label="Repeat for each year")
flowchart.edge('D', 'J', label="All completed years")
flowchart.edge('J', 'K')

# Sauvegarde et affichage
flowchart.render('flowchart_soil_estimation', view=True)
