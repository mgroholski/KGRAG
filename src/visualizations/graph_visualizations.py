import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph_topological(root_node):
    # Create a directed graph
    G = nx.DiGraph()

    # Use BFS to traverse and add nodes/edges
    visited = set()
    queue = [root_node]

    while queue:
        node = queue.pop(0)
        if node.value in visited:
            continue

        visited.add(node.value)
        G.add_node(node.value)

        for adj_node in node.adj:
            G.add_edge(node.value, adj_node.value)
            if adj_node.value not in visited:
                queue.append(adj_node)

    # Get topological generations (layers)
    topo_generations = list(nx.topological_generations(G))

    # Create positions for each node based on topological order
    pos = {}
    for i, generation in enumerate(topo_generations):
        for j, node in enumerate(generation):
            # Place nodes in topological order (left to right)
            # Center each generation vertically
            y_offset = (len(generation) - 1) / 2
            pos[node] = (i, y_offset - j)

    # Create the visualization
    plt.figure(figsize=(12, 8))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.5, arrowsize=20, edge_color='gray',
                          connectionstyle='arc3,rad=0.1')  # Curved edges

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    plt.axis('off')
    plt.title('Directed Graph - Topological Order')
    plt.tight_layout()
    plt.show()

    # return G
