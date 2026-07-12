import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from streamlit_agraph import agraph, Node, Edge, Config
from sklearn.neighbors import NearestNeighbors

from mantegna.code.data_getter import Data_Getter
from diffusion import prepare_diffusion, compute_diffusion

K = 3

st.set_page_config(layout="wide", page_title="Market Diffusion Simulation")

def get_continuous_color(value, cmap_name='plasma', vmax=0.1):
    try:
        val_clamped = float(min(value, vmax))
        norm = mcolors.Normalize(vmin=0.0, vmax=float(vmax))
        cmap = plt.get_cmap(cmap_name)
        rgba = cmap(norm(val_clamped))
        return mcolors.to_hex(rgba)
    except:
        return "#888888"

def draw_legend(vmax):
    st.sidebar.markdown("### 🌡️ Légende")
    gradient_css = "background: linear-gradient(to right, #0d0887, #cc4778, #f0f921);"
    legend_html = f"""
    <div style="width: 100%; font-family: sans-serif; font-size: 12px; color: #ddd; margin-bottom: 20px;">
        <div style="height: 15px; width: 100%; border-radius: 3px; {gradient_css} margin-bottom: 5px;"></div>
        <div style="display: flex; justify-content: space-between;">
            <span>Sain</span>
            <span>Infecté ({vmax*100:.0f}%)</span>
        </div>
    </div>
    """
    st.sidebar.markdown(legend_html, unsafe_allow_html=True)

def apply_anti_collision(pos, min_dist=60.0, iterations=30):
    """
    Pousse les noeuds trop proches (en pixels).
    """
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
        if not changed:
            break
    return {k: tuple(c) for k, c in zip(keys, coords)}


# --- 2. CALCULS ET STATE ---

# CONSTANTES FIGÉES (Vos réglages optimaux)
SPACING_K = 28.0
GLOBAL_SCALE = 2500

if 'G_knn' not in st.session_state:
    with st.spinner("Initialisation de la simulation..."):
        # A. Data
        getter = Data_Getter()
        if not getter.tickers: getter.get_tickers()
        getter.get_financials(getter.tickers, "2004-01-01", "2024-01-01")
        log_returns = getter.get_log_returns()
        tickers = list(log_returns.columns)
        
        # B. G_knn
        corr_matrix = log_returns.corr(min_periods=252).fillna(-1.0)
        distance_matrix_filled = np.sqrt(2 * (1 - corr_matrix)).fillna(2.0)
        knn = NearestNeighbors(n_neighbors= K + 1, metric='precomputed')
        knn.fit(distance_matrix_filled)
        distances, indices = knn.kneighbors(distance_matrix_filled)

        # 3. Construction du Graphe
        G_knn = nx.Graph()
        stock_names = distance_matrix_filled.columns

        for i in range(len(stock_names)):
            source = stock_names[i]
            for rank in range(1, K + 1):
                target = stock_names[indices[i][rank]]
                dist = distances[i][rank]
                G_knn.add_edge(source, target, weight=float(dist))
        
        # C. LAYOUT
        with st.spinner("Calcul du layout optimal..."):
    # On oublie le circular_layout qui force une forme ronde artificielle
            pos_kk = nx.kamada_kawai_layout(
                G_knn,
                scale=GLOBAL_SCALE,
                weight=None 
            )
            
            # D. MISE A L'ECHELLE ET SÉCURITÉ
            # Kamada-Kawai est déjà bien proportionné, on ajuste juste l'échelle
            pos_final = {k: (v[0], v[1]) for k, v in pos_kk.items()}
            
            # On garde l'anti-collision mais avec un rayon plus petit pour laisser 
            # les clusters se former naturellement
            pos_final = apply_anti_collision(pos_final, min_dist=40.0, iterations=20)
                
        # D. MISE A L'ECHELLE
        pos_final = {k: (v[0], v[1]) for k, v in pos_kk.items()}
            
        # E. ANTI-COLLISION
        pos_final = apply_anti_collision(pos_final, min_dist=60.0, iterations=30)

        # F. Sauvegarde
        st.session_state['G_knn'] = G_knn
        st.session_state['tickers'] = tickers
        st.session_state['pos'] = pos_final

G_knn = st.session_state['G_knn']
tickers = st.session_state['tickers']
pos = st.session_state['pos']

# --- 3. PARAMETRES DE SIMULATION ---
st.sidebar.title("Market Simulation")

default_idx = tickers.index("AAPL") if "AAPL" in tickers else 0
source_node = st.sidebar.selectbox("Patient Zéro", options=tickers, index=default_idx)

st.sidebar.subheader("Physique")
time_t = st.sidebar.slider("Temps (t)", 0.0, 100.0, 0.0, 0.1)
sigma = st.sidebar.slider("Conductivité (Sigma)", 0.1, 2.0, 0.5, 0.1)

st.sidebar.subheader("Visuel")
sensitivity = st.sidebar.slider("Sensibilité Visuelle", 0.01, 0.5, 0.05, 0.01)
show_labels = st.sidebar.checkbox("Afficher tous les noms", value=False)
draw_legend(sensitivity)

# Bouton de secours discret (au cas où le graphe bug au chargement)
if st.sidebar.button("Reset Graph", help="Recalculer le layout en cas de problème d'affichage"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

current_pos = st.session_state['pos']

if st.sidebar.checkbox("Layout Radial (Centré sur l'infection)", value=False):
    with st.spinner("Calcul du layout radial..."):
        # 1. Calcule la distance (nombre de sauts) de chaque noeud par rapport à la source
        levels = nx.single_source_shortest_path_length(G_knn, source_node)
        
        # 2. Organise les noeuds en listes par niveau de distance [[source], [voisins_1], [voisins_2]...]
        max_dist = max(levels.values()) if levels else 0
        nlist = [[n for n in G_knn.nodes() if levels.get(n) == i] for i in range(max_dist + 1)]
        
        # 3. Calcule le layout en cercles concentriques
        radial_pos = nx.shell_layout(G_knn, nlist=nlist)
        
        # 4. Mise à l'échelle pour l'affichage
        current_pos = {k: (v[0] * GLOBAL_SCALE, v[1] * GLOBAL_SCALE) for k, v in radial_pos.items()}

# --- 4. VISUALISATION ---
G_diff = prepare_diffusion(G_knn, sigma=sigma)
heat_vector = compute_diffusion(G_diff, time=time_t, start_node=source_node, tickers=tickers)

st.title(f"Simulation : {source_node}")

vis_nodes = []
vis_edges = []

for i, ticker in enumerate(tickers):
    if ticker not in current_pos: continue
    if ticker not in pos: continue
    
    heat = float(heat_vector[i])
    x, y = current_pos[ticker]
    color = get_continuous_color(heat, vmax=sensitivity)
    
    is_source = (ticker == source_node)
    is_hot = (heat > sensitivity * 0.1)
    
    if is_source:
        size = 25
        label =f'{ticker}'
        font_color = "#fff"
    elif is_hot or show_labels:
        size = 15 + (heat * 15)
        label = f'{ticker}'
        font_color = "#eee"
    else:
        size = 10
        label = f'{ticker}'
        font_color = "#555"
        
    vis_nodes.append(Node(
        id=ticker,
        label=label,
        size=size,
        color=color,
        x=x, y=y,
        shape="dot",
        font={'color': font_color, 'size': 14},
        borderWidth=1,
        borderColor="#222"
    ))

for u, v in G_knn.edges():
    vis_edges.append(Edge(source=u, target=v, color="#444", width=1.0))


config = Config(
    width="100%",
    height=800,
    directed=False, 
    physics=False, 
    hierarchical=False,
    interaction={"hover": True, "zoomView": True, "dragView": True} 
)

agraph(nodes=vis_nodes, edges=vis_edges, config=config)