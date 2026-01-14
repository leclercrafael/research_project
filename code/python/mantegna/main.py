import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os

import yfinance as yf

from data_getter import Data_Getter

start_date = '2004-01-01'
end_date = '2024-01-01'

#DATA IMPORTATION

#Import data 
data_getter = Data_Getter()

#Import tickers
sp500_tickers = data_getter.get_tickers()

#Import data (Adj Close)
data = data_getter.get_financials(tickers=sp500_tickers, start_date=start_date, end_date=end_date)

#Import log returns
log_returns = data_getter.get_log_returns()

tickers_sample = sp500_tickers[:] 
print(tickers_sample)

#CORRELATION AND DISTANCE

corr_matrix = log_returns.corr(min_periods=252) # 1 an de données minimum pour affirmer une corrélation
corr_matrix.fillna(0) # On met à 0 les données qui n'ont pas au moins 1 an en commun

# On a deux possibilités pour les distances, nous allons les définir ici :

# d(i,j) = sqrt(2 * (1 - rho_ij))

# d(i,j) =  1 - rho_ij**2


def plot_mantegna():
    #On va commencer par la première
    # d(i,j) = sqrt(2 * (1 - rho_ij))
    dist_matrix = np.sqrt(2 * (1 - corr_matrix))
    dist_matrix = dist_matrix.fillna(0) # Sécurité

    G = nx.from_pandas_adjacency(dist_matrix)
    mst = nx.minimum_spanning_tree(G)

    print(f"✅ Graphe final : {len(mst.nodes)} noeuds (Actions).")

    # 5. Visualisation
    print("🎨 Génération de l'image...")
    plt.figure(figsize=(16, 12))

    # Degré = importance du noeud
    degrees = dict(mst.degree())

    # Couleurs et Tailles
    node_sizes = [v * 15 + 30 for v in degrees.values()]
    node_colors = list(degrees.values())

    # Layout : Kamada-Kawai est top pour les arbres, sinon spring_layout
    try:
        pos = nx.kamada_kawai_layout(mst)
    except:
        pos = nx.spring_layout(mst, seed=42)

    # Dessin
    nx.draw_networkx_edges(mst, pos, alpha=0.2, edge_color='grey')
    nodes = nx.draw_networkx_nodes(mst, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.Spectral)

    # Labels pour les Hubs (> 6 connexions)
    hubs = {k: k for k, v in degrees.items() if v > 6}
    nx.draw_networkx_labels(mst, pos, labels=hubs, font_size=9, font_weight="bold")

    plt.colorbar(nodes, label="Connectivité (Degré)")
    plt.title("Baseline Mantegna : S&P 500 (2004-2024)\nApproche Inclusive (Données incomplètes acceptées)", fontsize=15)
    plt.axis('off')

    # Sauvegarde de l'image
    if not os.path.exists('code/graphs'):
        os.makedirs('code/graphs', exist_ok=True)
    plt.savefig('code/graphs/mantegna_baseline.png', dpi=300)
    print("Image sauvegardée dans code/graphs/mantegna_baseline.png")

    plt.show()

if __name__ == "__main__":
        plot_mantegna()



def plot_static_mantegna(start_date : str, end_date : str) -> None :
     pass

