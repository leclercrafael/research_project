import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
from pyvis.network import Network
import webbrowser 
import json
from sklearn.neighbors import NearestNeighbors

import yfinance as yf

from data_getter import Data_Getter

SECTOR_COLORS = {
    "Technology": "#1f77b4",             # Bleu
    "Healthcare": "#d62728",             # Rouge
    "Financial Services": "#2ca02c",     # Vert
    "Consumer Cyclical": "#ff7f0e",      # Orange
    "Communication Services": "#9467bd", # Violet
    "Industrials": "#8c564b",            # Marron
    "Consumer Defensive": "#bcbd22",     # Jaune/Olive
    "Energy": "#171717",                 # Noir
    "Utilities": "#17becf",              # Cyan
    "Real Estate": "#e377c2",            # Rose
    "Basic Materials": "#7f7f7f",        # Gris
    "Inconnu": "#d3d3d3"                 # Gris clair
}

class Mantegna:

    def __init__(self, start_date : str = '2004-01-01', end_date : str = '2024-01-01') -> None :
        self.start_date = start_date
        self.end_date = end_date

        self.data_getter = Data_Getter()
        self.tickers = self.data_getter.get_tickers()
        self.data = self.data_getter.get_financials(tickers = self.tickers, start_date=self.start_date, end_date=self.end_date)
        self.log_returns = self.data_getter.get_log_returns()

        self.corr_matrix = (self.log_returns.corr(min_periods=252)).fillna(0) # 1 an de données minimum pour affirmer une corrélation

        self.sectors_data = {}
        


        if self.data_getter.kg : 

            json_path = './json/kg_sectors.json'

            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.sectors_data = json.load(f)
            else:
                print(f"⚠️ Attention : '{json_path}' introuvable. Les secteurs seront marqués 'Inconnu'.")

        elif self.data_getter.get_adr:

            json_path = './json/sectors_snp+50adr.json'

            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.sectors_data = json.load(f)
            else:
                print(f"⚠️ Attention : '{json_path}' introuvable. Les secteurs seront marqués 'Inconnu'.")



    def plot_static_mantegna(self, distance : int = 1) -> None : 
        '''
        Start/End : intervall of time for financials data gathering
        Distance : choice of the used distance :
            - 1 : d(i,j) = sqrt(2 * (1 - rho_ij))
            - 2 : d(i,j) =  1 - rho_ij**2

        We already gathered the data during the init phase, so now we are going to us it !
        '''

        print(self.tickers)

        # Calculation of the distances matrix
        if distance == 1 : 
            self.dist_matrix = np.sqrt(2 * (1 - self.corr_matrix))
        elif distance == 2 :
            self.dist_matrix = 1-np.square(self.corr_matrix)
        else:
            print('Invalid distance')
            return None
            
        
        # Calculation of the topology 
        print('Topology calculation')
        self.G = nx.from_pandas_adjacency(self.dist_matrix)
        self.mst = nx.minimum_spanning_tree(self.G)   
        #We keep them in self because we arre going to use in the diffusion process

        #Calcultaion of the graph
        print('Layout calculation')
        self.pos = nx.kamada_kawai_layout(self.mst)


        #Generation of the interactive web page to see the graph

        print(" Vizualization generation")
    
        net = Network(height="100vh", width="100%", bgcolor="#222222", font_color="white", select_menu=True, cdn_resources='remote')
        
        options_string = """
        var options = {
          "nodes": { 
              "font": { "size": 16, "strokeWidth": 4, "strokeColor": "#000000" }, 
              "shape": "dot" 
          },
          "edges": {
              "color": { "inherit": true },
              "smooth": { "type": "continuous" },
              "width": 1
          },
          "physics": {
              "enabled": true,
              "barnesHut": {
                  "gravitationalConstant": -30000, 
                  "centralGravity": 0.3,
                  "springLength": 100,
                  "springConstant": 0.04,
                  "damping": 0.09,
                  "avoidOverlap": 1
              },
              "stabilization": {
                  "enabled": true,
                  "iterations": 1000,
                  "fit": true
              }
          },
          "interaction": {
              "navigationButtons": true,
              "keyboard": true,
              "zoomView": true,
              "dragNodes": true, 
              "hover": true
          }
        }
        """
        net.set_options(options_string)
        
        degrees = dict(self.mst.degree())
        SCALE_FACTOR = 2000 
        
        # Ajout des nœuds
        for node in self.mst.nodes():

            info = self.sectors_data.get(node, {})
            name = info.get("name", node)
            sector = info.get("sector", "Inconnu")

            deg = degrees[node]
            if deg > 10:
                color = "#ff4d4d" 
                size = 40
            elif deg > 4:
                color = "#ffae42" 
                size = 25
            else:
                color = "#97c2fc" 
                size = 15
                
            tooltip = f"{node} : {name}. Sector : {sector}. Connexions: {deg}"
            
            x_pos = self.pos[node][0] * SCALE_FACTOR
            y_pos = self.pos[node][1] * SCALE_FACTOR
            
            # Note : On ne met PAS physics=False ici, sinon l'anti-chevauchement (avoidOverlap) ne marchera pas au début.
            # On laisse le JS s'en occuper à la fin.
            net.add_node(node, label=node, title=tooltip, color=color, size=size, x=x_pos, y=y_pos)

        for source, target, data in self.mst.edges(data=True):
            net.add_edge(source, target, color="rgba(150,150,150,0.4)")

        # Sauvegarde
        output_path = os.path.abspath("./graphs/interactive_mst.html")
        if not os.path.exists('./graphs'):
            os.makedirs('./graphs')
        
        net.write_html(output_path)
        
        # --- INJECTION DU "FREEZE" ULTIME ---
        with open(output_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Ce script JS va s'exécuter une fois que la physique a fini de placer les bulles.
        # Il va parcourir TOUS les nœuds et leur dire : "Maintenant, tu ne bouges plus jamais (fixed=true)".
        js_injection = """
        <script type="text/javascript">
            network.on("stabilizationIterationsDone", function() {
                console.log("Stabilisation terminée -> Verrouillage des noeuds.");
                
                // 1. On coupe la physique globale
                network.setOptions( { physics: false } );
                
                // 2. On récupère tous les noeuds
                var nodes = network.body.data.nodes.get();
                
                // 3. On force la propriété 'fixed' à true pour chacun
                nodes.forEach(function(n) {
                    n.fixed = true;
                });
                
                // 4. On met à jour le dataset
                network.body.data.nodes.update(nodes);
                
                console.log("Noeuds verrouillés.");
            });
        </script>
        <style>
            #loadingBar { display: none !important; }
        </style>
        </head>
        """
        html_content = html_content.replace('</head>', js_injection)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("Ouverture du graphique...")
        webbrowser.open('file://' + output_path)







    def plot_cluster_mantegna(self, distance : int = 1) -> None :

        
        # Pour l'exemple, je remets juste les lignes critiques avant la génération :
        if distance == 1 : self.dist_matrix = np.sqrt(2 * (1 - self.corr_matrix))
        elif distance == 2 : self.dist_matrix = 1-np.square(self.corr_matrix)
        
        print('Topology calculation')
        self.G = nx.from_pandas_adjacency(self.dist_matrix)
        self.mst = nx.minimum_spanning_tree(self.G) 
    
        print('Layout calculation')
        self.pos = nx.kamada_kawai_layout(self.mst)

        print("Vizualization generation (Secteurs)")
        net = Network(height="100vh", width="100%", bgcolor="#000000", font_color="white", select_menu=True, cdn_resources='remote')
        
        # Options avec physique active au début pour l'espacement
        options_string = """
        var options = {
          "nodes": { 
              "font": { "size": 16, "strokeWidth": 4, "strokeColor": "#000000" }, 
              "shape": "dot" 
          },
          "edges": {
              "color": { "inherit": true },
              "smooth": { "type": "continuous" },
              "width": 1
          },
          "physics": {
              "enabled": true,
              "barnesHut": {
                  "gravitationalConstant": -30000, 
                  "centralGravity": 0.3,
                  "springLength": 95,
                  "avoidOverlap": 1
              },
              "stabilization": { "iterations": 1000, "fit": true }
          },
          "interaction": { "navigationButtons": true, "keyboard": true, "zoomView": true }
        }
        """
        net.set_options(options_string)
        
        degrees = dict(self.mst.degree())
        SCALE_FACTOR = 2000 
        
        for node in self.mst.nodes():
            # Infos du json
            info = self.sectors_data.get(node, {})
            name = info.get("name", node)
            sector = info.get("sector", "Inconnu")
            
            # Couleurs selon secteur
            color = SECTOR_COLORS.get(sector, "#d3d3d3")

            deg = degrees[node]
            size = 10 + (deg * 1.5) # Formule simple de taille

            tooltip = f"{node} : {name}. Secteur: {sector}. Connexions: {deg}"
            x_pos = self.pos[node][0] * SCALE_FACTOR
            y_pos = self.pos[node][1] * SCALE_FACTOR
            
            # Ajout du nœud avec le groupe 'sector' (utile pour la légende Pyvis native ou le filtrage)
            net.add_node(node, label=node, title=tooltip, color=color, size=size, x=x_pos, y=y_pos, group=sector)

        for source, target, data in self.mst.edges(data=True):
            net.add_edge(source, target, color="rgba(150,150,150,0.2)") # Arêtes plus discrètes

        # Sauvegarde
        output_path = os.path.abspath("./graphs/interactive_sectors.html")
        if not os.path.exists('./graphs'): os.makedirs('./graphs')
        net.write_html(output_path)
        
        # --- INJECTION DU FREEZE + LÉGENDE HTML ---
        with open(output_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Construction de la légende HTML
        legend_html = "<div id='legend' style='position: absolute; bottom: 20px; left: 20px; background-color: rgba(0,0,0,0.7); padding: 15px; border-radius: 10px; color: white; font-family: Arial; font-size: 12px; pointer-events: none;'>"
        legend_html += "<h3 style='margin-top:0;'>Secteurs</h3>"
        for sec, col in SECTOR_COLORS.items():
            if sec != "Inconnu": # On n'affiche pas inconnu si pas besoin
                legend_html += f"<div style='display: flex; align-items: center; margin-bottom: 5px;'><span style='width: 15px; height: 15px; background-color: {col}; border-radius: 50%; display: inline-block; margin-right: 10px;'></span>{sec}</div>"
        legend_html += "</div>"

        # Script JS combiné (Freeze + Injection CSS)
        custom_injection = f"""
        <script type="text/javascript">
            network.on("stabilizationIterationsDone", function() {{
                network.setOptions( {{ physics: false }} );
                var nodes = network.body.data.nodes.get();
                nodes.forEach(function(n) {{ n.fixed = true; }});
                network.body.data.nodes.update(nodes);
                console.log("Noeuds verrouillés par secteur.");
            }});
        </script>
        
        <style>
            #loadingBar {{ display: none !important; }}
        </style>
        
        {legend_html}
        </head>
        """
        
        html_content = html_content.replace('</head>', custom_injection)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("Ouverture du graphique sectoriel...")
        webbrowser.open('file://' + output_path)




    def get_graph_data(self, distance : int = 1) :

        if distance == 1 : self.dist_matrix = np.sqrt(2 * (1 - self.corr_matrix))
        elif distance == 2 : self.dist_matrix = 1-np.square(self.corr_matrix)

        print('Topology calculation')
        self.G = nx.from_pandas_adjacency(self.dist_matrix)
        self.mst = nx.minimum_spanning_tree(self.G) 

        print('Layout calculation')
        self.pos = nx.kamada_kawai_layout(self.mst)

        return self.mst, self.pos, self.sectors_data
    


    def plot_knn_mantegna(self, distance: int = 1, k: int = 3) -> None:


        # 1. Calcul de la matrice de distance (Mantegna)
        if distance == 1:
            self.dist_matrix = np.sqrt(2 * (1 - self.corr_matrix))
        elif distance == 2:
            self.dist_matrix = 1 - np.square(self.corr_matrix)

        # Nettoyage des NaNs (2.0 = distance max)
        distance_matrix_filled = self.dist_matrix.fillna(2.0)

        # 2. Algorithme KNN
        knn = NearestNeighbors(n_neighbors=k + 1, metric='precomputed')
        knn.fit(distance_matrix_filled)
        distances, indices = knn.kneighbors(distance_matrix_filled)

        # 3. Construction du Graphe
        G_knn = nx.Graph()
        stock_names = distance_matrix_filled.columns

        for i in range(len(stock_names)):
            source = stock_names[i]
            for rank in range(1, k + 1):
                target = stock_names[indices[i][rank]]
                dist = distances[i][rank]
                G_knn.add_edge(source, target, weight=float(dist))

        # 4. Calcul du Layout (KAMADA-KAWAI)
        # Contrairement au Spring, il évite l'effet "blob" central sur 600 noeuds
        print('Calcul du Layout (Kamada-Kawai)...')
        self.pos = nx.kamada_kawai_layout(G_knn, weight='weight')

        print("Génération de la visualisation Pyvis...")
        net = Network(height="100vh", width="100%", bgcolor="#222222", font_color="white", select_menu=True, cdn_resources='remote')

        # CONFIGURATION PYVIS (Physique désactivée)
        options_string = """
        var options = {
          "nodes": { 
              "font": { "size": 16, "strokeWidth": 4, "strokeColor": "#000000" }, 
              "shape": "dot" 
          },
          "edges": {
              "color": { "inherit": true },
              "smooth": { "type": "continuous" }
          },
          "physics": { "enabled": false },
          "interaction": { "navigationButtons": true, "zoomView": true }
        }
        """
        net.set_options(options_string)

        # Facteur d'échelle pour Kamada-Kawai (nécessite souvent une grande valeur)
        SCALE_FACTOR = 8000 
        degrees = dict(G_knn.degree())

        for node in G_knn.nodes():
            info = self.sectors_data.get(node, {})
            name = info.get("name", node)
            sector = info.get("sector", "Inconnu")
            color = SECTOR_COLORS.get(sector, "#d3d3d3")
            
            size = 10 + (degrees[node] * 2)
            
            # Positionnement manuel
            x_pos = self.pos[node][0] * SCALE_FACTOR
            y_pos = self.pos[node][1] * SCALE_FACTOR
            
            tooltip = f"Ticker : {node}. Name : {name}. Sector: {sector}"
            net.add_node(node, label=node, title=tooltip, color=color, size=size, x=x_pos, y=y_pos, group=sector)

        for source, target, data in G_knn.edges(data=True):
            net.add_edge(source, target, color="rgba(150,150,150,0.2)", width=1)

        # Sauvegarde
        output_path = os.path.abspath("./graphs/interactive_knn_sectors.html")
        if not os.path.exists('./graphs'): os.makedirs('./graphs')
        net.write_html(output_path)

        # 5. Injection Légende & Suppression du Loading Bar
        with open(output_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        legend_html = "<div id='legend' style='position: absolute; bottom: 20px; left: 20px; background-color: rgba(0,0,0,0.8); padding: 15px; border-radius: 10px; color: white; font-family: Arial; font-size: 12px; pointer-events: none; z-index: 9999;'>"
        legend_html += "<h3 style='margin-top:0;'>Secteurs</h3>"
        for sec, col in SECTOR_COLORS.items():
            if sec != "Inconnu":
                legend_html += f"<div style='display: flex; align-items: center; margin-bottom: 5px;'><span style='width: 12px; height: 12px; background-color: {col}; border-radius: 50%; display: inline-block; margin-right: 10px;'></span>{sec}</div>"
        legend_html += "</div>"

        # On cache proprement la barre de chargement par CSS
        custom_injection = f"""
        <style>
            #loadingBar {{ display: none !important; }}
            .vis-navigation {{ background: rgba(255,255,255,0.1); border-radius: 5px; }}
        </style>
        {legend_html}
        </head>
        """
        html_content = html_content.replace('</head>', custom_injection)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Graphique sauvegardé : {output_path}")
        webbrowser.open('file://' + output_path)