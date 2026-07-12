import networkx as nx
import numpy as np
import scipy

def prepare_diffusion(graph : nx.Graph, sigma : float = 0.1) -> nx.Graph :
    
    diffusion_graph = graph.copy()

    print('MST Graph -> Diffusion Graph')

    for u, v, data in diffusion_graph.edges(data=True):

        distance = data['weight']
        conductance = np.exp(-(distance**2) / (sigma**2))
        diffusion_graph[u][v]['weight'] = conductance
        diffusion_graph[u][v]['distance'] = distance

    return diffusion_graph


def compute_diffusion(graph : nx.Graph, time : float, start_node : str, tickers : list) :

    L = nx.laplacian_matrix(graph, nodelist=tickers).toarray()

    H_t = scipy.linalg.expm(-L*time)
    H_t = np.real(H_t)

    start_node_idx = tickers.index(start_node)

    chaleur_reçue = H_t[start_node_idx, :]

    return chaleur_reçue

        

