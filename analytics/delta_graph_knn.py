import streamlit as st
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from streamlit_agraph import agraph, Node, Edge, Config
from sklearn.neighbors import NearestNeighbors
import os
import sys

# --- CONFIGURATION ET IMPORTS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kg.code.FinKGBuilder import FinKGBuilder
from mantegna.code.data_getter import Data_Getter
from mantegna.code.diffusion import prepare_diffusion, compute_diffusion

st.set_page_config(layout="wide", page_title="Financial Risk Lab (k-NN Delta)")

# --- CONSTANTES ---
K_NEIGHBORS = 3  # Nombre de voisins pour le graphe de marché
GLOBAL_SCALE = 2500

# --- HELPER FUNCTIONS ---
def normalize_heat(raw_vector):
    """Second-Rank Normalization : Evite que la source (1.0) n'écrase les voisins."""
    u_sorted = np.sort(np.unique(raw_vector))
    # On prend l'avant dernier max (le voisin le plus chaud) comme référence 1.0
    v_max = u_sorted[-2] if len(u_sorted) > 1 else raw_vector.max()
    return np.clip(raw_vector / (v_max + 1e-10), 0, 1)

def apply_anti_collision(pos, min_dist=60.0, iterations=30):
    """Évite que les noeuds ne se chevauchent visuellement."""
    keys = list(pos.keys())
    n = len(keys)
    coords = np.array([pos[k] for k in keys], dtype=float)
    for _ in range(iterations):
        changed = False
        for i in range(n):
            diff = coords - coords[i]
            dist_sq = np.sum(diff**2, axis=1)
            too_close = (dist_sq < min_dist**2) & (dist_sq > 0)
            if np.any(too_close):
                conflict_vectors = diff[too_close]
                repulsion = -conflict_vectors 
                move = np.mean(repulsion, axis=0)
                norm = np.linalg.norm(move)
                if norm > 0:
                    move = move / norm * (min_dist * 0.1)
                    coords[i] += move
                    changed = True
        if not changed: break
    return {k: tuple(c) for k, c in zip(keys, coords)}

# --- LOADERS ---
@st.cache_resource
def get_kg_manager():
    manager = FinKGBuilder()
    manager.load_data()
    manager.build_matrix()
    return manager

builder = get_kg_manager()

# --- INITIALISATION k-NN MARKET GRAPH ---
if 'G_knn' not in st.session_state:
    with st.spinner("Construction du graphe de marché (k-NN)..."):
        # 1. Data Getter
        getter = Data_Getter()
        if not getter.tickers: getter.get_tickers()
        getter.get_financials(getter.tickers, "2004-01-01", "2024-01-01")
        log_returns = getter.get_log_returns()
        tickers_market = list(log_returns.columns)
        
        # 2. Distance Matrix
        corr_matrix = log_returns.corr(min_periods=252).fillna(0)
        dist_matrix = np.sqrt(2 * (1 - corr_matrix)).fillna(2.0)
        
        # 3. k-NN Algorithm
        knn = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, metric='precomputed')
        knn.fit(dist_matrix)
        distances, indices = knn.kneighbors(dist_matrix)
        
        # 4. Construction NetworkX
        G_knn = nx.Graph()
        stock_names = dist_matrix.columns
        for i in range(len(stock_names)):
            source = stock_names[i]
            for rank in range(1, K_NEIGHBORS + 1): # On skip rank 0 (soi-même)
                target = stock_names[indices[i][rank]]
                dist = distances[i][rank]
                G_knn.add_edge(source, target, weight=float(dist))
        
        # 5. Layout Kamada-Kawai + Anti-Collision
        pos_kk = nx.kamada_kawai_layout(G_knn, scale=GLOBAL_SCALE, weight=None)
        pos_final = {k: (v[0], v[1]) for k, v in pos_kk.items()}
        pos_final = apply_anti_collision(pos_final, min_dist=50.0, iterations=20)

        st.session_state['G_knn'] = G_knn
        st.session_state['tickers'] = tickers_market
        st.session_state['pos'] = pos_final

G_knn = st.session_state['G_knn']
tickers = st.session_state['tickers']
pos = st.session_state['pos']

# --- SIDEBAR CONTROL ---
st.sidebar.title("Simulation Control (k-NN)")

# Sélecteur de Mode
mode = st.sidebar.radio("Mode de Visualisation", 
                        ["Market (Correlation)", "Knowledge Graph (Structural)", "Divergence (Delta KG-Market)"])

# Sélecteur Patient Zéro
default_idx = tickers.index("NVDA") if "NVDA" in tickers else 0
source_node = st.sidebar.selectbox("Patient Zéro", options=tickers, index=default_idx)

# Paramètres Physiques
st.sidebar.subheader("Physique de Diffusion")
time_t = st.sidebar.slider("Temps (t)", 0.0, 10.0, 1.2, 0.1)
sigma = st.sidebar.slider("Conductivité (Sigma)", 0.1, 5.0, 0.5, 0.1)
show_labels = st.sidebar.checkbox("Afficher Tickers", value=False)

# --- MOTEUR DE DIFFUSION ---

# 1. Diffusion Market (sur topology k-NN)
# Important : on prépare la diffusion avec le kernel gaussien sur les poids k-NN
g_diff = prepare_diffusion(G_knn, sigma=sigma) 
h_market_raw = compute_diffusion(g_diff, time=time_t, start_node=source_node, tickers=tickers)
h_market_norm = normalize_heat(h_market_raw)
map_market = dict(zip(tickers, h_market_norm))

# 2. Diffusion KG (Structurel)
if source_node in builder.tickers:
    kg_idx = builder.tickers.index(source_node)
    # Note : on scale le temps KG pour qu'il matche l'échelle du marché
    h_kg_raw = builder.compute_heat(kg_idx, time=time_t * sigma) 
    h_kg_norm = normalize_heat(h_kg_raw)
    map_kg = dict(zip(builder.tickers, h_kg_norm))
else:
    map_kg = {t: 0.0 for t in tickers}

# --- RENDU GRAPHIQUE ---
st.title(f"Visualisation : {mode} (Topology k-NN)")

if mode == "Divergence (Delta KG-Market)":
    st.info("🔴 Rouge : Risque Structurel Caché (KG > Market) | 🔵 Bleu : Bruit de Marché (Market > KG)")
    cmap = plt.get_cmap('RdBu_r') 
else:
    cmap = plt.get_cmap('inferno')

vis_nodes = []
# On parcourt les tickers
for ticker in tickers:
    if ticker not in pos: continue
    
    val_kg = float(map_kg.get(ticker, 0.0))
    val_market = float(map_market.get(ticker, 0.0))
    
    # Logique de Couleur Delta
    if mode == "Market (Correlation)":
        heat = val_market
        color = mcolors.to_hex(cmap(heat))
        label_threshold = 0.4
    elif mode == "Knowledge Graph (Structural)":
        heat = val_kg
        color = mcolors.to_hex(cmap(heat))
        label_threshold = 0.4
    else: # DELTA
        heat = val_kg - val_market # [-1, 1]
        color = mcolors.to_hex(cmap((heat + 1) / 2)) # Shift vers [0, 1] pour la colormap
        label_threshold = 0.3 # Plus sensible pour le Delta

    is_source = (ticker == source_node)
    
    # Styles des noeuds
    if is_source:
        size = 35
        shape = "star"
        label = f"★ {ticker}"
        node_color = "#FF0000" # Toujours rouge vif pour la source
        font_color = "#000"
    else:
        size = 12 + (abs(heat) * 20)
        shape = "dot"
        # On affiche le label si c'est chaud ou si c'est demandé
        label = ticker if (abs(heat) > label_threshold or show_labels) else ""
        node_color = color
        font_color = "#000" if (mode != "Delta" and heat > 0.6) else "#555"

    vis_nodes.append(Node(
        id=ticker, label=label, size=size, color=node_color, 
        x=pos[ticker][0], y=pos[ticker][1], shape=shape,
        font={'color': font_color, 'size': 14},
        borderWidth=1, borderColor="#333"
    ))

# Edges : On affiche les liens k-NN (plus dense que le MST)
vis_edges = [
    Edge(source=u, target=v, color="#ccc", width=0.5, opacity=0.2) 
    for u, v in G_knn.edges()
]

config = Config(
    width="100%", height=2000, 
    physics=False, # On fige le layout pré-calculé
    interaction={"hover": True, "zoomView": True}
)

agraph(nodes=vis_nodes, edges=vis_edges, config=config)