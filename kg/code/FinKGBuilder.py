import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
import networkx as nx
from pyvis.network import Network
import os
import json
import webbrowser 
from sklearn.neighbors import NearestNeighbors


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


class FinKGBuilder ():

    def __init__(self) -> None:
        
        self.DATA_DIR = './FinDKG_dataset'

        self.redirection_map = {}
        self.global_mx_index = {}

        self.relation_weights = {
                                    0 : 0.95,
                                    1: 0.95,
                                    2: 0.95,
                                    3: 0.80,
                                    4: 0.75,
                                    5: 0.95,
                                    6: 0.80,
                                    7: 0.50,
                                    8: 0.60,
                                    9: 0.50,
                                    10: 0.20,
                                    11: 0.55,
                                    12: 0.20,
                                    13: 0.20,
                                    14: 0.45
                                }
        
        self.W = None

        self.sectors_data = {}
    
        json_path = './json/kg_sectors.json'

        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                self.sectors_data = json.load(f)
        else:
            print(f"⚠️ Attention : '{json_path}' introuvable. Les secteurs seront marqués 'Inconnu'.")


    def load_data (self) -> None:

        df = pd.read_csv(f'{self.DATA_DIR}/kg_companies.txt', sep=': ')
        df['Name'] = df['Name'].str.replace('"', '', regex=False)
        df['Tickers'] = df['Tickers'].str.replace('"', '', regex=False).str.replace(',', '', regex=False)
        
        df_entities = pd.read_csv(f'{self.DATA_DIR}/entity2id.txt', sep='\t', header=None, names=['name', 'id', 'x', 'y'])
        

        # On crée ici nos dictionnaires de mapping afin de pouvoir faire les liens entre les ids !

        self.tickers = df['Tickers'].unique().tolist()
        self.name_to_id = dict(zip(df_entities['name'], df_entities['id']))
        self.id_to_name = dict(zip(df_entities['id'], df_entities['name']))

        tickers_to_companies = {}
        tickers_to_companies_id = {}
        for ticker in self.tickers:
            tickers_to_companies[ticker]=df[df['Tickers']==ticker]['Name'].tolist()
            tickers_to_companies_id[ticker] = []
            for name in df[df['Tickers']==ticker]['Name'].tolist():
                tickers_to_companies_id[ticker].append(self.name_to_id[name])

        self.main_ids_companies = [tickers_to_companies_id[ticker][0] for ticker in self.tickers]

        tickers_to_main_ids = {}
        main_ids_to_tickers = {}
        for i in range(len(self.tickers)):
            tickers_to_main_ids[self.tickers[i]]=self.main_ids_companies[i]
            main_ids_to_tickers[self.main_ids_companies[i]]=self.tickers[i]        


        # Dans la redirection_map on arrive à mapper tous les ids vers là ils doivent pointer afin de merge les noeuds
        self.redirection_map = {}

        for ticker, ids in tickers_to_companies_id.items():
            main_id = tickers_to_main_ids[ticker]
            for company_id in ids:
                self.redirection_map[company_id]=main_id

        for id in df_entities.id :
            if id not in self.redirection_map:
                self.redirection_map[id]=id


        companies_id = []
        for ids in tickers_to_companies_id.values():
            companies_id.extend(ids)

        not_companies_id = list(set(self.redirection_map.keys()) - set(companies_id))

        mx_orders_id = self.main_ids_companies + not_companies_id

        #On peut enfin avoir le mapping entre l'index des entreprises avec le merge et leur place dans la matrice des poids
        self.global_mx_index = {id_kg: i for i, id_kg in enumerate(mx_orders_id)}


    def build_matrix(self) -> None:

        row=[]
        col=[]
        weight=[]

        files = ['/train.txt', '/test.txt', '/valid.txt']
   
        for file in files:

            current_file = self.DATA_DIR + file
            df = pd.read_csv(current_file, sep='\s+', header=None, names=['h', 'r', 't', 'ts','id'])

            for h, r, t, ts, i in df.values:
                h, r, t = int(h), int(r), int(t)

        # Ici on peut mettre une condition sur le timestamp pour éviter le leakage

                main_h = self.redirection_map[h]
                main_t = self.redirection_map[t]

                if main_h != main_t:

                    mx_h = self.global_mx_index[main_h]
                    mx_t = self.global_mx_index[main_t]

                    value = self.relation_weights[r]

                    row.append(mx_h)
                    row.append(mx_t)
                    col.append(mx_t)
                    col.append(mx_h)
                    weight.append(value)
                    weight.append(value)




        N = len(self.global_mx_index)
        W_brute = sp.csr_matrix((weight, (row, col)), shape=(N, N))

        # Normalisation :
        degrees = np.array(W_brute.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1e-10
        d_inv_sqrt = 1.0 / np.sqrt(degrees)
        D_inv_sqrt = sp.diags(d_inv_sqrt)


        self.W = D_inv_sqrt @ W_brute @ D_inv_sqrt


    def compute_heat(self, start_ticker_idx, time : float) -> list:

        # 1. On crée le vecteur initial (V0)
        N = self.W.shape[0]
        v0 = np.zeros(N)
        v0[start_ticker_idx] = 1.0
        
        # Calcul du Laplacien
        I = sp.eye(N)
        L = I - self.W
        
        # Diffusion 
        vt = expm_multiply(-L * time, v0)
        
        return vt[:438]
    

    def plot_mst_cluster(self) -> None :

        W_dense_commonalities = (self.W @ self.W)[:438, :438].toarray()
    
        # 2. On s'assure que la matrice est symétrique et sans auto-boucles
        np.fill_diagonal(W_dense_commonalities, 0)
        
        # 3. Création du graphe complet
        G_proj = nx.from_numpy_array(W_dense_commonalities)
        
        # 4. Conversion en distance (pour que le MST connecte les plus proches)
        # On utilise 1 / (poids + epsilon) pour éviter la division par 0
        for u, v, d in G_proj.edges(data=True):
            d['distance'] = 1.0 / (d['weight'] + 1e-6)
        
        # 5. MST
        mst_graph = nx.minimum_spanning_tree(G_proj, weight='distance')
        
        # 6. Mapping des noms
        mapping = {i: ticker for i, ticker in enumerate(self.tickers)}
        self.mst = nx.relabel_nodes(mst_graph, mapping)

    
        print('Layout calculation')
        self.pos = nx.kamada_kawai_layout(self.mst)

        print("Vizualization generation (Secteurs)")
        net = Network(height="100vh", width="100%", bgcolor="#222222", font_color="white", select_menu=True, cdn_resources='remote')
        
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
        output_path = os.path.abspath("./graphs/kg_interactive_sectors.html")
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


    def plot_knn_cluster(self, distance: int = 1, k: int = 3) -> None:


        similarity_matrix = (self.W @ self.W)[:438, :438].toarray()
        np.fill_diagonal(similarity_matrix, 0)
        
        G_knn = nx.Graph()
        G_knn.add_nodes_from(self.tickers)
        
        for i, ticker in enumerate(self.tickers):
            row = similarity_matrix[i]
            nearest_indices = np.argsort(row)[-k:][::-1]
            
            for neighbor_idx in nearest_indices:
                weight = row[neighbor_idx]
                if weight > 0: # On n'ajoute un lien que s'il y a une connexion réelle
                    G_knn.add_edge(ticker, self.tickers[neighbor_idx], weight=weight)

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
            
            tooltip = f"<b>{node}</b><br>{name}<br>Secteur: {sector}"
            net.add_node(node, label=node, title=tooltip, color=color, size=size, x=x_pos, y=y_pos, group=sector)

        for source, target, data in G_knn.edges(data=True):
            net.add_edge(source, target, color="rgba(150,150,150,0.2)", width=1)

        # Sauvegarde
        output_path = os.path.abspath("./graphs/kg_interactive_knn_sectors.html")
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