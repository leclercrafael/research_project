import numpy as np
import pandas as pd
import networkx as nx
from pyvis.network import Network
import os
import webbrowser 
from data_getter import Data_Getter
import json

with open("code/sp500_full.json", "r", encoding="utf-8") as f:
    sp500 = json.load(f)

def run_interactive_mantegna():
    print("🚀 Chargement des données...")
    getter = Data_Getter()
    if not getter.tickers:
        getter.get_tickers()
    
    # 1. Données
    log_returns = getter.get_financials(getter.tickers, "2004-01-01", "2024-01-01")
    log_returns = getter.get_log_returns()
    
    # 2. Calculs 
    print("🧮 Calcul de la topologie...")
    corr_matrix = log_returns.corr(min_periods=252).fillna(0)
    dist_matrix = np.sqrt(2 * (1 - corr_matrix)).fillna(0)
    
    G_complete = nx.from_pandas_adjacency(dist_matrix)
    mst = nx.minimum_spanning_tree(G_complete)
    
    # --- CHANGEMENT ICI : KAMADA-KAWAI ---
    print("📐 Calcul du layout Kamada-Kawai (cela peut prendre quelques secondes)...")
    # Cet algo place les points pour que la distance visuelle corresponde à la distance mathématique
    pos = nx.kamada_kawai_layout(mst)

    # 3. Visualisation Interactive
    print("🎨 Génération du site web interactif...")
    
    net = Network(height="100vh", width="100%", bgcolor="#222222", font_color="white", select_menu=True, cdn_resources='remote')
    
    options_string = """
    var options = {
      "nodes": { 
        "font": { "size": 16, "strokeWidth": 4, "strokeColor": "#000000" }, 
        "shape": "dot" 
      },
      "edges": {
        "color": { "inherit": true },
        "smooth": false,
        "width": 1
      },
      "physics": {
        "enabled": false,
        "stabilization": true
      },
      "interaction": {
        "navigationButtons": true,
        "keyboard": true,
        "zoomView": true 
      }
    }
    """
    net.set_options(options_string)
    
    degrees = dict(mst.degree())
    
    # ECHELLE ENCORE PLUS GRANDE POUR KAMADA
    # Kamada a tendance à faire des dessins très "sphériques", il faut beaucoup d'espace
    SCALE_FACTOR = 5000 
    
    for node in mst.nodes():
        deg = degrees[node]
        if deg > 10:
            color = "#ff4d4d" # Rouge (Hubs)
            size = 50
        elif deg > 4:
            color = "#ffae42" # Orange
            size = 30
        else:
            color = "#97c2fc" # Bleu
            size = 15
            
        tooltip = f"{node} : Connexions: {deg}. Name : {sp500[f'{node}']['name']}. Description : {sp500[f'{node}']['description']}"
        
        # Positionnement
        x_pos = pos[node][0] * SCALE_FACTOR
        y_pos = pos[node][1] * SCALE_FACTOR
        
        net.add_node(node, label=node, title=tooltip, color=color, size=size, x=x_pos, y=y_pos)

    for source, target, data in mst.edges(data=True):
        net.add_edge(source, target, color="rgba(150,150,150,0.4)")

    # 4. Sauvegarde
    output_path = os.path.abspath("code/graphs/interactive_mst.html")
    if not os.path.exists('code/graphs'):
        os.makedirs('code/graphs')
    
    net.write_html(output_path)
    
    # Patch CSS
    with open(output_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    style_injection = """
    <style>
        #loadingBar { display: none !important; }
    </style>
    </head>
    """
    html_content = html_content.replace('</head>', style_injection)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("🌍 Ouverture du graphique...")
    webbrowser.open('file://' + output_path)

if __name__ == "__main__":
    run_interactive_mantegna()