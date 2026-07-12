import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import os
import sys

# Gestion des chemins
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.abspath(__file__))
mantegna_path = os.path.join(current_dir, '..', 'mantegna', 'code')
kg_path = os.path.join(current_dir, '..', 'kg', 'code')

sys.path.append(mantegna_path)
sys.path.append(kg_path)

from mantegna.code.Mantegna import Mantegna
from kg.code.FinKGBuilder import FinKGBuilder

def calculate_sector_homophily(G: nx.Graph, sectors_data: dict) -> float:
    """ Calcule le % moyen de voisins partageant le même secteur. """
    node_scores = []
    for node in G.nodes():
        info = sectors_data.get(node, {})
        my_sector = info.get("sector", "Unknown")
        
        if my_sector in ["Unknown", "Inconnu"]:
            continue
            
        neighbors = list(G.neighbors(node))
        k = len(neighbors)
        if k == 0: continue
            
        same_sector_count = sum(1 for n in neighbors if sectors_data.get(n, {}).get("sector", "Unknown") == my_sector)
        node_scores.append(same_sector_count / k)
    
    return np.mean(node_scores) if node_scores else 0.0

def build_fast_mst_market(market_instance):
    """ Construit le MST du marché sans calculer le Layout (Gain de temps) """
    dist_matrix = np.sqrt(2 * (1 - market_instance.corr_matrix))
    G = nx.from_pandas_adjacency(dist_matrix)
    return nx.minimum_spanning_tree(G)

def build_fast_mst_kg(kg_instance):
    """ Construit le MST du KG sans calculer le Layout ni ouvrir le navigateur """
    # Réplication de la logique mathématique de FinKGBuilder.plot_mst_cluster
    # Projection : W * W (Similarité)
    W_dense = (kg_instance.W @ kg_instance.W)[:438, :438].toarray()
    np.fill_diagonal(W_dense, 0)
    
    G_proj = nx.from_numpy_array(W_dense)
    
    # Inversion pour MST (Distance = 1/Poids)
    for u, v, d in G_proj.edges(data=True):
        d['distance'] = 1.0 / (d['weight'] + 1e-6)
        
    mst_raw = nx.minimum_spanning_tree(G_proj, weight='distance')
    
    # Remapping des IDs (0, 1, 2) vers Tickers (AAPL, NVDA...)
    mapping = {i: ticker for i, ticker in enumerate(kg_instance.tickers)}
    return nx.relabel_nodes(mst_raw, mapping)

def rebuild_mantegna_knn(mantegna_instance, k=5):
    """ KNN Marché Rapide """
    dist_matrix = np.sqrt(2 * (1 - mantegna_instance.corr_matrix)).fillna(2.0)
    knn = NearestNeighbors(n_neighbors=k + 1, metric='precomputed')
    knn.fit(dist_matrix)
    distances, indices = knn.kneighbors(dist_matrix)
    
    G = nx.Graph()
    stock_names = dist_matrix.columns
    for i in range(len(stock_names)):
        source = stock_names[i]
        for rank in range(1, k + 1):
            target = stock_names[indices[i][rank]]
            G.add_edge(source, target)
    return G

def rebuild_kg_knn(kg_instance, k=5):
    """ KNN KG Rapide """
    similarity_matrix = (kg_instance.W @ kg_instance.W)[:438, :438].toarray()
    np.fill_diagonal(similarity_matrix, 0)
    
    G = nx.Graph()
    G.add_nodes_from(kg_instance.tickers)
    
    for i, ticker in enumerate(kg_instance.tickers):
        row = similarity_matrix[i]
        nearest_indices = np.argsort(row)[-k:][::-1]
        for neighbor_idx in nearest_indices:
            if row[neighbor_idx] > 0:
                G.add_edge(ticker, kg_instance.tickers[neighbor_idx])
    return G

def run_analysis():
    print("1. Chargement des données (Patience pour YFinance)...")
    market = Mantegna() 
    kg = FinKGBuilder()
    kg.load_data()
    kg.build_matrix()
    
    sectors = market.sectors_data 
    
    print("2. Calcul des MST (Mode Mathématiques Pures)...")
    # On utilise nos fonctions rapides au lieu des méthodes de classe qui plot
    mst_market = build_fast_mst_market(market)
    mst_kg = build_fast_mst_kg(kg)
    
    score_mst_market = calculate_sector_homophily(mst_market, sectors)
    score_mst_kg = calculate_sector_homophily(mst_kg, sectors)
    
    print(f"   -> MST Market Homophily : {score_mst_market:.2%}")
    print(f"   -> MST KG Homophily     : {score_mst_kg:.2%}")


    print("3. Calcul des KNN (k=3)...")
    k_val = 3
    G_knn_market = rebuild_mantegna_knn(market, k=k_val)
    G_knn_kg = rebuild_kg_knn(kg, k=k_val)
    
    score_knn_market = calculate_sector_homophily(G_knn_market, sectors)
    score_knn_kg = calculate_sector_homophily(G_knn_kg, sectors)
    
    print(f"   -> KNN Market Homophily : {score_knn_market:.2%}")
    print(f"   -> KNN KG Homophily     : {score_knn_kg:.2%}")

    print("\n--- RÉSULTATS POUR LE RAPPORT ---")
    print(f"The Market KNN (k={k_val}) exhibits a Sector Homophily of {score_knn_market:.1%},")
    print(f"while the Knowledge Graph KNN shows {score_knn_kg:.1%}.")
    
    if score_knn_market > score_knn_kg:
        print("This confirms that market correlations are sector-biased, whereas the KG captures cross-industry structural links.")

        
    print("3. Calcul des KNN (k=4)...")
    k_val = 4
    G_knn_market = rebuild_mantegna_knn(market, k=k_val)
    G_knn_kg = rebuild_kg_knn(kg, k=k_val)
    
    score_knn_market = calculate_sector_homophily(G_knn_market, sectors)
    score_knn_kg = calculate_sector_homophily(G_knn_kg, sectors)
    
    print(f"   -> KNN Market Homophily : {score_knn_market:.2%}")
    print(f"   -> KNN KG Homophily     : {score_knn_kg:.2%}")

    print("\n--- RÉSULTATS POUR LE RAPPORT ---")
    print(f"The Market KNN (k={k_val}) exhibits a Sector Homophily of {score_knn_market:.1%},")
    print(f"while the Knowledge Graph KNN shows {score_knn_kg:.1%}.")
    
    if score_knn_market > score_knn_kg:
        print("This confirms that market correlations are sector-biased, whereas the KG captures cross-industry structural links.")


    
    print("3. Calcul des KNN (k=5)...")
    k_val = 5
    G_knn_market = rebuild_mantegna_knn(market, k=k_val)
    G_knn_kg = rebuild_kg_knn(kg, k=k_val)
    
    score_knn_market = calculate_sector_homophily(G_knn_market, sectors)
    score_knn_kg = calculate_sector_homophily(G_knn_kg, sectors)
    
    print(f"   -> KNN Market Homophily : {score_knn_market:.2%}")
    print(f"   -> KNN KG Homophily     : {score_knn_kg:.2%}")

    print("\n--- RÉSULTATS POUR LE RAPPORT ---")
    print(f"The Market KNN (k={k_val}) exhibits a Sector Homophily of {score_knn_market:.1%},")
    print(f"while the Knowledge Graph KNN shows {score_knn_kg:.1%}.")


    
    if score_knn_market > score_knn_kg:
        print("This confirms that market correlations are sector-biased, whereas the KG captures cross-industry structural links.")


    print("3. Calcul des KNN (k=10)...")
    k_val = 10
    G_knn_market = rebuild_mantegna_knn(market, k=k_val)
    G_knn_kg = rebuild_kg_knn(kg, k=k_val)
    
    score_knn_market = calculate_sector_homophily(G_knn_market, sectors)
    score_knn_kg = calculate_sector_homophily(G_knn_kg, sectors)
    
    print(f"   -> KNN Market Homophily : {score_knn_market:.2%}")
    print(f"   -> KNN KG Homophily     : {score_knn_kg:.2%}")

    print("\n--- RÉSULTATS POUR LE RAPPORT ---")
    print(f"The Market KNN (k={k_val}) exhibits a Sector Homophily of {score_knn_market:.1%},")
    print(f"while the Knowledge Graph KNN shows {score_knn_kg:.1%}.")

    print("3. Calcul des KNN (k=20)...")
    k_val = 20
    G_knn_market = rebuild_mantegna_knn(market, k=k_val)
    G_knn_kg = rebuild_kg_knn(kg, k=k_val)
    
    score_knn_market = calculate_sector_homophily(G_knn_market, sectors)
    score_knn_kg = calculate_sector_homophily(G_knn_kg, sectors)
    
    print(f"   -> KNN Market Homophily : {score_knn_market:.2%}")
    print(f"   -> KNN KG Homophily     : {score_knn_kg:.2%}")

    print("\n--- RÉSULTATS POUR LE RAPPORT ---")
    print(f"The Market KNN (k={k_val}) exhibits a Sector Homophily of {score_knn_market:.1%},")
    print(f"while the Knowledge Graph KNN shows {score_knn_kg:.1%}.")

if __name__ == "__main__":
    run_analysis()