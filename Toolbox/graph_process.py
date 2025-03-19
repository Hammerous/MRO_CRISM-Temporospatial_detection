import networkx as nx

def intsec2graph(intsec_lst, min_val):
    # non-directed graph
    G = nx.Graph(intsec_lst)
    # Find all maximal cliques
    cliques = list(nx.find_cliques(G))
    
    # Filter and group cliques by size using a standard dictionary
    clique_dict = {}
    for clique in cliques:
        if len(clique) >= min_val:  # Only consider cliques with at least 3 nodes
            size = len(clique)
            if size not in clique_dict:
                clique_dict[size] = []
            clique_dict[size].append(clique)
    
    # Yield cliques in ascending order of size
    for size in sorted(clique_dict.keys()):
        yield size, clique_dict[size]

def find_weighted_degrees(edges):
    # non-directed graph
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    weighted_degrees = {node: sum(data['weight'] for _, _, data in G.edges(node, data=True)) for node in G.nodes()}
    return weighted_degrees
