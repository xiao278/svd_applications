import networkx
import matplotlib.cm

def get_mappings(G:networkx.Graph):
    '''returns (n2i, i2n)'''
    node_to_index = {}
    index_to_node = {}

    for i, node in enumerate(G.nodes):
        node_to_index[node] = i
        index_to_node[i] = node
    return (node_to_index, index_to_node)

def bipartite_clustered_positioning(G:networkx.Graph, INDEX:list[int]):
    (top_nodes, bottom_nodes) = networkx.algorithms.bipartite.sets(G)
    (node_to_index, index_to_node) = get_mappings(G)

    top_node_index = {}
    bottom_node_index = {}
    top_index_node = {}
    bottom_index_node = {}

    for node in G.nodes:
        index = node_to_index[node]
        if node in top_nodes:
            top_node_index[node] = index
            top_index_node[index] = node
        elif node in bottom_nodes:
            bottom_node_index[node] = index
            bottom_index_node[index] = node
        else:
            assert False

    top_node_order = {}
    bottom_node_order = {}

    for index in INDEX:
        node = index_to_node[index]
        if index in top_index_node:
            top_node_order[node] = len(top_node_order)
        elif index in bottom_index_node:
            bottom_node_order[node] = len(bottom_node_order)
        else:
            assert False

    x_gap = 8
    y_gap = 1

    pos = {}

    for node, order in top_node_order.items():
        pos[node] = (0, y_gap * order)

    for node, order in bottom_node_order.items():
        pos[node] = (x_gap, y_gap * order)

    return pos

def spiral_clustered_positioning(G:networkx.Graph, INDEX:list[int], resolution=0.35):
    (node_to_index, index_to_node) = get_mappings(G)
    reordered_graph = networkx.Graph()
    new_nodes = []
    for index in INDEX:
        new_nodes.append(index_to_node[index])
    reordered_graph.add_nodes_from(new_nodes)
    reordered_graph.add_edges_from(G.edges())
    pos = networkx.spiral_layout(reordered_graph, resolution=resolution)
    return pos


def get_colormappings(G:networkx.Graph, INDEX:list[int], clusters:list[dict], cmap_name='Set3'):
    (node_to_index, index_to_node) = get_mappings(G)
    group_labels = {}
    for node in G.nodes:
        group_labels[node] = 0
    for i in range(len(clusters)):
        for index_of_INDEX in clusters[i]:
            index = INDEX[index_of_INDEX]
            node = index_to_node[index]
            group_labels[node] = i + 1
    unique_groups = set(group_labels.values())
    colors = matplotlib.cm.get_cmap(cmap_name, len(unique_groups))
    color_map = {group: colors(i) for i, group in enumerate(unique_groups)} # color assigned to each group label
    node_colors = ['lightgray' if group_labels[node] == 0 else color_map[group_labels[node]] for node in G.nodes]
    edge_colors = []
    for u, v in G.edges():
        u_group = group_labels[u]
        v_group = group_labels[v]
        if u_group != 0 and v_group != 0 and u_group == v_group:
            edge_colors.append(color_map[u_group])
        else:
            edge_colors.append(None)

    return (node_colors, edge_colors)