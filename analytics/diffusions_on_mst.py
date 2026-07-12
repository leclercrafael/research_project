import streamlit as st
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from streamlit_agraph import agraph, Node, Edge, Config
import os
import sys
import json

# --- CONFIGURATION ET IMPORTS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kg.code.FinKGBuilder import FinKGBuilder
from mantegna.code.data_getter import Data_Getter
from mantegna.code.diffusion import prepare_diffusion, compute_diffusion

st.set_page_config(layout="wide", page_title="Financial Risk Lab")

@st.cache_resource
def get_kg_manager():
    manager = FinKGBuilder()
    manager.load_data()
    manager.build_matrix()
    return manager

builder = get_kg_manager()

# --- INITIALISATION MST ---
if 'mst' not in st.session_state:
    with st.spinner("Initialisation du graphe..."):
        getter = Data_Getter()
        if not getter.tickers: getter.get_tickers()
        getter.get_financials(getter.tickers, "2004-01-01", "2024-01-01")
        log_returns = getter.get_log_returns()
        tickers_market = list(log_returns.columns)
        corr_matrix = log_returns.corr(min_periods=252).fillna(0)
        dist_matrix = np.sqrt(2 * (1 - corr_matrix)).fillna(0)
        G_complete = nx.from_pandas_adjacency(dist_matrix)
        mst = nx.minimum_spanning_tree(G_complete)
        pos_kk = nx.kamada_kawai_layout(mst, scale=2500, weight=None)
        st.session_state['mst'] = mst
        st.session_state['tickers'] = tickers_market
        st.session_state['pos'] = pos_kk

mst = st.session_state['mst']
tickers = st.session_state['tickers']
pos = st.session_state['pos']

# --- SIDEBAR ---
st.sidebar.title("Simulation Control")
# AJOUT DU MODE DELTA
mode = st.sidebar.radio("Mode de Visualisation", 
                        ["Market (Correlation)", "Knowledge Graph (Structural)", "Divergence (Delta KG-Market)"])

source_node = st.sidebar.selectbox("Patient Zéro", options=tickers, index=tickers.index("NVDA") if "NVDA" in tickers else 0)

st.sidebar.subheader("Physique")
time_t = st.sidebar.slider("Temps (t)", 0.0, 10.0, 1.2, 0.1)
sigma = st.sidebar.slider("Conductivité (Sigma)", 0.1, 5.0, 1.5, 0.1)
show_labels = st.sidebar.checkbox("Afficher Tickers", value=False)

# --- FONCTION DE NORMALISATION INTERNE ---
def normalize_heat(raw_vector):
    u_sorted = np.sort(np.unique(raw_vector))
    v_max = u_sorted[-2] if len(u_sorted) > 1 else raw_vector.max()
    return np.clip(raw_vector / (v_max + 1e-10), 0, 1)

# --- CALCUL DES VECTEURS ---
# 1. Calcul Market
g_diff = prepare_diffusion(mst, sigma=sigma)
h_market_raw = compute_diffusion(g_diff, time=time_t, start_node=source_node, tickers=tickers)
h_market_norm = normalize_heat(h_market_raw)
map_market = dict(zip(tickers, h_market_norm))

# 2. Calcul KG
if source_node in builder.tickers:
    kg_idx = builder.tickers.index(source_node)
    h_kg_raw = builder.compute_heat(kg_idx, time=time_t * sigma)
    h_kg_norm = normalize_heat(h_kg_raw)
    map_kg = dict(zip(builder.tickers, h_kg_norm))
else:
    map_kg = {t: 0.0 for t in tickers}

# --- RENDU ---
st.title(f"Visualisation : {mode}")

if mode == "Divergence (Delta KG-Market)":
    st.info("🔴 Rouge : Risque KG > Marché (Sous-estimé) | 🔵 Bleu : Risque Marché > KG (Sentiment)")
    cmap = plt.get_cmap('RdBu_r') # Red-Blue divergente
else:
    cmap = plt.get_cmap('inferno')

vis_nodes = []
for ticker in tickers:
    if ticker not in pos: continue
    
    val_kg = float(map_kg.get(ticker, 0.0))
    val_market = float(map_market.get(ticker, 0.0))
    
    # Sélection de la valeur selon le mode
    if mode == "Market (Correlation)":
        heat = val_market
        color = mcolors.to_hex(cmap(heat))
    elif mode == "Knowledge Graph (Structural)":
        heat = val_kg
        color = mcolors.to_hex(cmap(heat))
    else: # MODE DELTA
        heat = val_kg - val_market # Range de -1 à 1
        # On mappe [-1, 1] vers [0, 1] pour la colormap
        color = mcolors.to_hex(cmap((heat + 1) / 2))

    is_source = (ticker == source_node)
    
    # Taille et labels
    if is_source:
        size, shape, label, node_color = 35, "star", f"★ {ticker}", "#FF0000"
    else:
        size = 12 + (abs(heat) * 15)
        shape = "dot"
        label = ticker if (abs(heat) > 0.2 or show_labels) else ""
        node_color = color

    vis_nodes.append(Node(
        id=ticker, label=label, size=size, color=node_color, 
        x=pos[ticker][0], y=pos[ticker][1], shape=shape,
        font={'color': '#000' if (mode != "Delta" and heat > 0.6) else '#888', 'size': 14}
    ))

vis_edges = [Edge(source=u, target=v, color="#eee", width=0.5, opacity=0.3) for u, v in mst.edges()]
config = Config(width="100%", height=850, physics=False, interaction={"hover": True, "zoomView": True})
agraph(nodes=vis_nodes, edges=vis_edges, config=config)