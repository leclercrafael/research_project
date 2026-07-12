import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pyvis.network import Network
import os

# Vos imports
from mantegna.code.data_getter import Data_Getter
from diffusion import prepare_diffusion, compute_diffusion

st.set_page_config(layout="wide", page_title="Market Diffusion Sim")

# --- 1. FONCTIONS UTILITAIRES ---
def get_continuous_color(value, cmap_name='plasma', vmax=0.1):
    # Conversion sécurisée
    try:
        val_clamped = float(min(value, vmax))
        norm = mcolors.Normalize(vmin=0.0, vmax=float(vmax))
        cmap = plt.get_cmap(cmap_name)
        rgba = cmap(norm(val_clamped))
        return mcolors.to_hex(rgba)
    except:
        return "#ffffff"

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

# --- 2. GESTION DES DONNÉES ET DU LAYOUT (SESSION STATE) ---
# On utilise le session_state pour stocker le layout UNE FOIS POUR TOUTES.
# Ça évite que le graphe ne bouge d'un millimètre quand on touche au slider.

if 'mst' not in st.session_state:
    with st.spinner("Initialisation des données et du layout..."):
        # A. Chargement
        getter = Data_Getter()
        if not getter.tickers:
            getter.get_tickers()
        getter.get_financials(getter.tickers, "2004-01-01", "2024-01-01")
        log_returns = getter.get_log_returns()
        tickers = list(log_returns.columns)
        
        # B. Calcul MST
        corr_matrix = log_returns.corr(min_periods=252).fillna(0)
        dist_matrix = np.sqrt(2 * (1 - corr_matrix)).fillna(0)
        G_complete = nx.from_pandas_adjacency(dist_matrix)
        mst = nx.minimum_spanning_tree(G_complete)
        
        # C. CALCUL DU LAYOUT (Kamada Kawai est le meilleur pour les arbres)
        # Il respecte la hiérarchie des clusters (Secteurs).
        pos = nx.kamada_kawai_layout(mst)
        
        # D. ÉTIREMENT DES COORDONNÉES (LE FIX CRUCIAL)
        # NetworkX donne des coords entre -1 et 1.
        # On les multiplie par 4000 pour avoir un graphe de 8000 pixels de large.
        scaled_pos = {}
        scale_factor = 4000 
        
        for node, coords in pos.items():
            # On stocke en float pur pour éviter les bugs JSON
            scaled_pos[node] = (float(coords[0]) * scale_factor, float(coords[1]) * scale_factor)

        # E. Sauvegarde dans la session
        st.session_state['mst'] = mst
        st.session_state['tickers'] = tickers
        st.session_state['pos'] = scaled_pos

# Récupération depuis la session (Instantané)
mst = st.session_state['mst']
tickers = st.session_state['tickers']
pos = st.session_state['pos']


# --- 3. PARAMÈTRES INTERACTIFS ---
st.sidebar.title("Risk Simulation")

# Sélecteurs
default_idx = tickers.index("AAPL") if "AAPL" in tickers else 0
source_node = st.sidebar.selectbox("Patient Zéro", options=tickers, index=default_idx)

st.sidebar.subheader("Physique")
time_t = st.sidebar.slider("Temps (t)", 0.0, 10.0, 2.0, 0.1)
sigma = st.sidebar.slider("Conductivité (Sigma)", 0.1, 2.0, 0.5, 0.1)

st.sidebar.subheader("Visuel")
sensitivity = st.sidebar.slider("Sensibilité", 0.01, 0.5, 0.05, 0.01)
show_labels = st.sidebar.checkbox("Afficher tous les noms", value=False)

draw_legend(sensitivity)

# --- 4. CALCUL DIFFUSION ---
# On recalcule juste les couleurs, pas les positions !
G_diff = prepare_diffusion(mst, sigma=sigma)
heat_vector = compute_diffusion(G_diff, time=time_t, start_node=source_node, tickers=tickers)

# --- 5. RENDU PYVIS ---
st.title(f"Simulation : {source_node}")

# Utilisation de 'remote' pour éviter l'écran blanc
net = Network(height="800px", width="100%", bgcolor="#111111", font_color="white", select_menu=True, cdn_resources='remote')

# CONFIGURATION STRICTE
# 1. Physics = False (Car on fournit des coordonnées 'x' et 'y' précises)
# 2. Scaling des noeuds pour qu'ils soient lisibles
options_string = """
var options = {
  "nodes": { 
      "shape": "dot",
      "font": { "size": 14, "color": "white" }
  },
  "edges": {
      "color": { "inherit": true, "opacity": 0.2 },
      "width": 1,
      "smooth": false
  },
  "physics": { 
      "enabled": false 
  },
  "interaction": { 
      "hover": true, 
      "zoomView": true,
      "dragNodes": false 
  }
}
"""
net.set_options(options_string)

degrees = dict(mst.degree())

for i, ticker in enumerate(tickers):
    if ticker not in mst.nodes(): continue
    if ticker not in pos: continue
    
    # Données
    heat = float(heat_vector[i])
    x, y = pos[ticker] # Ces coordonnées sont déjà x4000
    
    # Couleur
    color = get_continuous_color(heat, vmax=sensitivity)
    
    # Logique d'apparence (Noeuds petits pour éviter la bouillie)
    is_source = (ticker == source_node)
    is_hot = (heat > sensitivity * 0.1)
    
    if is_source:
        size = 40
        label = f"★ {ticker}"
        color = "#ffffff"
    elif is_hot or show_labels:
        size = 15 + (heat * 30) # Grossit si chaud
        label = ticker
    else:
        size = 10 # Petite taille fixe pour les noeuds sains
        label = " " # On cache le texte pour aérer

    tooltip = f"{ticker}. Chaleur: {heat:.4f}"
    
    # Ajout au graphe avec coordonnées fixes
    net.add_node(ticker, label=label, title=tooltip, color=color, size=size, x=x, y=y)

for u, v in mst.edges():
    net.add_edge(u, v, color="rgba(100,100,100,0.2)")

# --- 6. AFFICHAGE FINAL ---
try:
    path = os.path.join(os.getcwd(), "tmp_diff.html")
    net.write_html(path)
    
    # Injection CSS pour s'assurer que le canvas prend toute la place
    with open(path, 'r', encoding='utf-8') as f:
        html = f.read()
        html = html.replace('</head>', 
        """
        <style>
            body { margin: 0; padding: 0; }
            #mynetwork { width: 100%; height: 800px; border: 1px solid #333; }
            #loadingBar { display: none !important; }
        </style>
        </head>
        """)
        
    components.html(html, height=820, scrolling=False)

except Exception as e:
    st.error(f"Erreur d'affichage : {e}")