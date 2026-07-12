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

st.set_page_config(layout="wide", page_title="Financial Risk Lab (Rank-Based)")

# --- CONSTANTES ---
K_NEIGHBORS = 4  # Un peu plus de voisins pour lisser le marché
GLOBAL_SCALE = 2500

# --- FONCTION DE RANKING ---
def get_rankings(heat_vector):
    """
    Transforme un vecteur de chaleur en vecteur de Rangs.
    Le noeud le plus chaud a le rang 1. Le plus froid a le rang N.
    """
    # argsort donne les indices du plus petit au plus grand.
    # On veut le plus chaud (grand) en premier -> on inverse [::-1]
    sorted_indices = np.argsort(heat_vector)[::-1]
    
    # On crée un tableau de rangs vide
    ranks = np.empty_like(sorted_indices)
    
    # On remplit : ranks[index_du_noeud] = son_classement (0-indexed)
    ranks[sorted_indices] = np.arange(len(heat_vector))
    
    return ranks

# --- LOADERS (Inchangé) ---
@st.cache_resource
def get_kg_manager():
    manager = FinKGBuilder()
    manager.load_data()
    manager.build_matrix()
    return manager

builder = get_kg_manager()

if 'G_knn' not in st.session_state:
    with st.spinner("Construction du graphe de marché (k-NN)..."):
        getter = Data_Getter()
        if not getter.tickers: getter.get_tickers()
        getter.get_financials(getter.tickers, "2004-01-01", "2024-01-01")
        log_returns = getter.get_log_returns()
        tickers_market = list(log_returns.columns)
        
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
                dist = distances[i][rank]
                G_knn.add_edge(source, target, weight=float(dist))
        
        pos_kk = nx.kamada_kawai_layout(G_knn, scale=GLOBAL_SCALE, weight=None)
        st.session_state['G_knn'] = G_knn
        st.session_state['tickers'] = tickers_market
        st.session_state['pos'] = {k: (v[0], v[1]) for k, v in pos_kk.items()}

G_knn = st.session_state['G_knn']
tickers = st.session_state['tickers']
pos = st.session_state['pos']

# --- SIDEBAR ---
st.sidebar.title("Rank-Based Simulation")

mode = st.sidebar.radio("Mode de Visualisation", 
                        ["Market Heat", "KG Heat", "Divergence (Rank Delta)"])

default_idx = tickers.index("NVDA") if "NVDA" in tickers else 0
source_node = st.sidebar.selectbox("Patient Zéro", options=tickers, index=default_idx)

st.sidebar.subheader("Paramètres")
time_t = st.sidebar.slider("Temps de Diffusion", 0.0, 10.0, 2.5, 0.1)
sigma = st.sidebar.slider("Conductivité", 0.1, 5.0, 1.0, 0.1)
top_k_display = st.sidebar.slider("Afficher Top N Divergences", 5, 50, 10)

# --- CALCULS ---
source_idx = tickers.index(source_node) if source_node in tickers else 0

# 1. Diffusion Market
g_diff = prepare_diffusion(G_knn, sigma=sigma) 
h_market_raw = compute_diffusion(g_diff, time=time_t, start_node=source_node, tickers=tickers)
# -> RANKING MARKET (Petit rang = Chaud)
ranks_market = get_rankings(h_market_raw)

# 2. Diffusion KG
if source_node in builder.tickers:
    kg_idx = builder.tickers.index(source_node)
    h_kg_raw = builder.compute_heat(kg_idx, time=time_t * sigma) 
    
    # Mapping KG -> Tickers
    full_kg_heat = dict(zip(builder.tickers, h_kg_raw))
    aligned_kg_heat = np.array([full_kg_heat.get(t, 0.0) for t in tickers])
    
    # -> RANKING KG (Petit rang = Chaud)
    ranks_kg = get_rankings(aligned_kg_heat)
else:
    ranks_kg = np.full(len(tickers), len(tickers)) # Tout le monde est dernier

# --- VISUALISATION ---
st.title(f"Analyse par Rangs : {source_node}")

if mode == "Divergence (Rank Delta)":
    # Delta = Rang_Market - Rang_KG
    # Exemple : Market=100 (Froid), KG=5 (Chaud) -> Delta = +95 (ROUGE / DANGER)
    # Exemple : Market=5 (Chaud), KG=100 (Froid) -> Delta = -95 (BLEU / BRUIT)
    delta_ranks = ranks_market - ranks_kg
    
    # Normalisation pour la couleur (-N à +N vers 0-1)
    max_delta = len(tickers) / 2 # Facteur d'échelle arbitraire pour le contraste
    norm = mcolors.Normalize(vmin=-max_delta, vmax=max_delta)
    cmap = plt.get_cmap('RdBu_r')
    
    st.info(f"Analyse : Les nœuds Rouges sont proches structurellement (KG) mais ignorés par le marché.")

vis_nodes = []

# Pour éviter d'afficher 400 points gris, on filtre souvent sur ceux qui ont un delta significatif
# ou qui sont chauds dans au moins l'un des deux.

for i, ticker in enumerate(tickers):
    if ticker not in pos: continue
    
    r_m = ranks_market[i]
    r_k = ranks_kg[i]
    
    is_source = (ticker == source_node)
    
    if mode == "Divergence (Rank Delta)":
        val = delta_ranks[i]
        color = mcolors.to_hex(cmap(norm(val)))
        
        # Logique d'affichage intelligente
        # On affiche le label si c'est une grosse divergence (Top K positif ou négatif)
        is_interesting = (r_k < 20) or (r_m < 20) # Top 20 dans l'un des deux mondes
        
        if is_source:
            size, label = 35, f"★ {ticker}"
            color = "#FF0000"
        elif is_interesting:
            size = 15 + (abs(val) / len(tickers) * 20)
            # Label montre : Ticker (Rang KG vs Rang Mkt)
            label = f"{ticker}\n(K#{r_k} vs M#{r_m})"
        else:
            size = 5
            label = ""
            color = "#cccccc" # Gris discret pour le bruit de fond
            
    else: # Modes simples
        val = h_market_raw[i] if "Market" in mode else aligned_kg_heat[i]
        # Simple heatmap
        color = mcolors.to_hex(plt.get_cmap('inferno')(val / (np.max(val)+1e-9)))
        label = ticker if val > np.max(val)*0.1 else ""
        size = 10 + (val/np.max(val))*20

    vis_nodes.append(Node(
        id=ticker, label=label, size=size, color=color, 
        x=pos[ticker][0], y=pos[ticker][1], shape="dot" if not is_source else "star",
        font={'color': 'black', 'size': 12},
        borderWidth=1, borderColor="#333"
    ))

vis_edges = [Edge(source=u, target=v, color="#ccc", width=0.5, opacity=0.1) for u, v in G_knn.edges()]

config = Config(width="100%", height=850, physics=False, interaction={"hover": True, "zoomView": True})
agraph(nodes=vis_nodes, edges=vis_edges, config=config)