import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.neighbors import NearestNeighbors
import os
import sys

# --- CONFIGURATION (Chemins) ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kg.code.FinKGBuilder import FinKGBuilder
from mantegna.code.data_getter import Data_Getter
from mantegna.code.diffusion import prepare_diffusion, compute_diffusion

# --- PARAMETRES ---
SOURCE_NODE = "UPS"  # Ton cas d'usage
K_NEIGHBORS = 4
SIGMA = 1.0
TIME_T = 2.5
OUTPUT_FILE = "figure_ups_delta.pdf" # Export PDF pour LaTeX

# --- 1. RECONSTRUCTION DU GRAPHE (Copier-Coller de ta logique) ---
print("Chargement des données...")
getter = Data_Getter()
if not getter.tickers: getter.get_tickers()
getter.get_financials(getter.tickers, "2004-01-01", "2024-01-01")
log_returns = getter.get_log_returns()
tickers = list(log_returns.columns)

# k-NN
corr_matrix = log_returns.corr(min_periods=252).fillna(0)
dist_matrix = np.sqrt(2 * (1 - corr_matrix)).fillna(2.0)
knn = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, metric='precomputed')
knn.fit(dist_matrix)
distances, indices = knn.kneighbors(dist_matrix)

G_knn = nx.Graph()
stock_names = dist_matrix.columns
for i in range(len(stock_names)):
    source = stock_names[i]
    for rank in range(1, K_NEIGHBORS + 1):
        target = stock_names[indices[i][rank]]
        G_knn.add_edge(source, target, weight=float(distances[i][rank]))

# Positions (Fixes pour reproductibilité)
print("Calcul du Layout...")
pos = nx.kamada_kawai_layout(G_knn, scale=1.0) 

# --- 2. CALCUL DES DIFFUSIONS ---
print("Calcul de la diffusion...")
builder = FinKGBuilder()
builder.load_data(); builder.build_matrix()

# Market Diffusion & Rank
g_diff = prepare_diffusion(G_knn, sigma=SIGMA) 
h_market = compute_diffusion(g_diff, time=TIME_T, start_node=SOURCE_NODE, tickers=tickers)
ranks_market = np.argsort(np.argsort(h_market)[::-1])

# KG Diffusion & Rank
kg_idx = builder.tickers.index(SOURCE_NODE)
h_kg = builder.compute_heat(kg_idx, time=TIME_T * SIGMA)
full_kg_heat = dict(zip(builder.tickers, h_kg))
aligned_kg = np.array([full_kg_heat.get(t, 0.0) for t in tickers])
ranks_kg = np.argsort(np.argsort(aligned_kg)[::-1])

# Delta Rank
delta_ranks = ranks_market - ranks_kg # Positive = KG plus chaud (Rang plus petit)

# --- 3. DESSIN MATPLOTLIB (Haute Qualité) ---
print("Génération de la figure PDF...")
plt.figure(figsize=(15, 15)) # Grande taille
ax = plt.gca()

# Colormap
cmap = plt.get_cmap('RdBu_r')
norm = mcolors.Normalize(vmin=-len(tickers)/2, vmax=len(tickers)/2)

# Dessin des Edges (Gris très clair, en arrière plan)
nx.draw_networkx_edges(G_knn, pos, alpha=0.1, edge_color='#cccccc', width=0.5)

# Préparation des Noeuds
node_colors = []
node_sizes = []
labels = {}

for i, ticker in enumerate(tickers):
    val = delta_ranks[i]
    color = cmap(norm(val))
    
    is_source = (ticker == SOURCE_NODE)
    
    # Logique de filtrage (Top 20 ou Source)
    r_k = ranks_kg[i]
    r_m = ranks_market[i]
    is_interesting = (r_k < 20) or (r_m < 20) or is_source
    
    if is_source:
        node_colors.append('#FF0000') # Rouge vif
        node_sizes.append(800)
        labels[ticker] = ticker
    elif is_interesting:
        node_colors.append(color)
        node_sizes.append(300)
        labels[ticker] = ticker # Juste le ticker pour que ce soit lisible
    else:
        node_colors.append('#e0e0e0') # Gris clair pour le bruit
        node_sizes.append(30) # Tout petit

# Dessin des Noeuds
# 1. Le bruit de fond (petits gris)
nx.draw_networkx_nodes(G_knn, pos, nodelist=[t for i, t in enumerate(tickers) if node_sizes[i] == 30], 
                       node_color='#e0e0e0', node_size=30, alpha=0.5)

# 2. Les noeuds intéressants (Colorés)
interesting_indices = [i for i, size in enumerate(node_sizes) if size > 30]
interesting_tickers = [tickers[i] for i in interesting_indices]
interesting_colors = [node_colors[i] for i in interesting_indices]
interesting_sizes = [node_sizes[i] for i in interesting_indices]

nx.draw_networkx_nodes(G_knn, pos, nodelist=interesting_tickers, 
                       node_color=interesting_colors, node_size=interesting_sizes, 
                       edgecolors='black', linewidths=0.5)

# 3. Les Labels (avec repousse pour éviter le chevauchement si possible)
# Pour Matplotlib simple, on pose juste le texte
text_items = nx.draw_networkx_labels(G_knn, pos, labels=labels, font_size=10, font_weight='bold')

# Petit hack pour mettre un fond blanc sous le texte pour lisibilité
if text_items:
    for t in text_items.values():
        t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

plt.title(f"Delta-Rank Analysis: {SOURCE_NODE}", fontsize=20)
plt.axis('off')

# Sauvegarde
plt.tight_layout()
plt.savefig(OUTPUT_FILE, format='pdf', dpi=300)
print(f"Terminé ! Image sauvegardée sous : {OUTPUT_FILE}")
plt.show()