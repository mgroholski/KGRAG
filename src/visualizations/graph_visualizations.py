import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph_with_cycle_detection(root_node):
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

    # Check for cycles in the graph
    cycles = list(nx.simple_cycles(G))
    has_cycles = len(cycles) > 0

    # Determine layout - use topological generations if no cycles, or spring layout otherwise
    if not has_cycles:
        try:
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
        except nx.NetworkXUnfeasible:
            # This shouldn't happen since we already checked for cycles
            pos = nx.spring_layout(G, seed=42)
    else:
        # Use spring layout for graphs with cycles
        pos = nx.spring_layout(G, seed=42)

    # Create the visualization
    plt.figure(figsize=(12, 8))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')

    # Prepare edge lists - regular edges and cycle edges
    cycle_edges = []
    if has_cycles:
        for cycle in cycles:
            for i in range(len(cycle)):
                cycle_edges.append((cycle[i], cycle[(i+1) % len(cycle)]))

    regular_edges = [(u, v) for u, v in G.edges() if (u, v) not in cycle_edges]

    # Draw regular edges
    nx.draw_networkx_edges(G, pos, edgelist=regular_edges, width=1.5,
                          arrowsize=20, edge_color='gray',
                          connectionstyle='arc3,rad=0.1')

    # Draw cycle edges with different color
    if has_cycles:
        nx.draw_networkx_edges(G, pos, edgelist=cycle_edges, width=2.5,
                              arrowsize=20, edge_color='red',
                              connectionstyle='arc3,rad=0.1')

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    # Add a title
    if has_cycles:
        plt.title(f'Directed Graph with Cycles (Found {len(cycles)} cycles)')
        # Add a legend for cycles
        cycle_text = "Cycles found:\n"
        for i, cycle in enumerate(cycles):
            cycle_path = " → ".join([str(node) for node in cycle + [cycle[0]]])
            cycle_text += f"{i+1}. {cycle_path}\n"
        plt.figtext(0.5, 0.01, cycle_text, ha="center", fontsize=10,
                   bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    else:
        plt.title('Directed Graph - Topological Order (No Cycles)')

    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return G, cycles

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
